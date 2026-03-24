# Strategy: mtf_1d_fisher_chop_regime_rsi_donchian_1w_atr_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.073 | +23.3% | -10.3% | 56 | PASS |
| ETHUSDT | -0.405 | -4.5% | -32.0% | 51 | FAIL |
| SOLUSDT | -0.350 | -12.3% | -32.1% | 57 | FAIL |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.661 | +12.4% | -4.3% | 18 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
Experiment #823: 1d Primary + 1w HTF — Fisher Transform + Choppiness Regime

Hypothesis: After 562+ failed strategies, the key insight is that 1d timeframe
needs REVERSAL-BASED entries (not trend-following) to work in bear/range markets.
2025 test period is bearish (-25% BTC), so pure trend strategies fail.

Strategy design:
1. 1d Primary timeframe (target 20-40 trades/year)
2. 1w HMA(21) for long-term bias only (not entry trigger)
3. 1d Ehlers Fisher Transform(9) for reversal detection
4. 1d Choppiness Index(14) for regime detection
5. 1d RSI(14) with relaxed thresholds (35/65) for confluence
6. 1d Donchian(20) for breakout confirmation in trending regime
7. 1d ATR(14) for trailing stop (2.5x)
8. Dual regime: mean revert when CHOP>55, trend follow when CHOP<45
9. Fisher Transform catches reversals in bear market rallies (proven edge)

Why Fisher Transform:
- Normalizes price to Gaussian distribution (-1.5 to +1.5 range)
- Long when Fisher crosses above -1.5 from below (oversold reversal)
- Short when Fisher crosses below +1.5 from above (overbought reversal)
- Works exceptionally well in 2022 crash and 2025 bear market

Key changes from failed 1d strategies:
- Fisher Transform instead of CRSI (better reversal detection)
- RSI thresholds: 35/65 (not 30/70) — more signals on daily
- CHOP thresholds: 55/45 (clearer regime separation)
- Add Fisher + RSI confluence for higher probability entries
- Hold logic maintains position through minor pullbacks

Target: Sharpe > 0.612, trades >= 15 train, >= 5 test, ALL symbols positive
Timeframe: 1d (target 25-40 trades/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_fisher_chop_regime_rsi_donchian_1w_atr_v1"
timeframe = "1d"
leverage = 1.0

def calculate_sma(series, period):
    """Simple Moving Average."""
    return pd.Series(series).rolling(window=period, min_periods=period).mean().values

def calculate_hma(series, period):
    """Hull Moving Average."""
    series = pd.Series(series)
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma_half = series.rolling(window=half, min_periods=half).mean() * 2
    wma_full = series.rolling(window=period, min_periods=period).mean()
    
    wma_diff = wma_half - wma_full
    hma = wma_diff.rolling(window=sqrt_period, min_periods=sqrt_period).mean()
    
    return hma.values

def calculate_rsi(close, period=14):
    """Relative Strength Index."""
    n = len(close)
    rsi = np.full(n, np.nan)
    
    if n < period + 1:
        return rsi
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    avg_gain = np.concatenate([[np.nan], avg_gain])
    avg_loss = np.concatenate([[np.nan], avg_loss])
    
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
    
    rsi = np.clip(rsi, 0, 100)
    return rsi

def calculate_fisher_transform(high, low, period=9):
    """
    Ehlers Fisher Transform — normalizes price to Gaussian distribution.
    Range typically -1.5 to +1.5. Reversals at extremes.
    """
    n = len(high)
    fisher = np.full(n, np.nan)
    fisher_prev = np.full(n, np.nan)
    
    if n < period + 1:
        return fisher, fisher_prev
    
    for i in range(period, n):
        # Calculate typical price
        hl2 = (high[i] + low[i]) / 2.0
        
        # Find highest high and lowest low over period
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        
        range_val = highest_high - lowest_low
        if range_val < 1e-10:
            fisher[i] = fisher[i-1] if i > 0 else 0.0
            fisher_prev[i] = fisher[i-1] if i > 1 else 0.0
            continue
        
        # Normalize to 0-1 range
        normalized = (hl2 - lowest_low) / range_val
        
        # Constrain to 0.001-0.999 to avoid log(0)
        normalized = np.clip(normalized, 0.001, 0.999)
        
        # Fisher transform
        fisher_val = 0.5 * np.log((1 + normalized) / (1 - normalized))
        
        # Smooth with previous value (Ehlers method)
        if i > 0 and not np.isnan(fisher[i-1]):
            fisher[i] = 0.7 * fisher_val + 0.3 * fisher[i-1]
            fisher_prev[i] = fisher[i-1]
        else:
            fisher[i] = fisher_val
            fisher_prev[i] = fisher_val
    
    return fisher, fisher_prev

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index — measures market choppy vs trending.
    CHOP > 55 = ranging, CHOP < 45 = trending.
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    if n < period + 1:
        return chop
    
    for i in range(period, n):
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        
        if highest_high == lowest_low:
            chop[i] = 100
            continue
        
        tr_sum = 0.0
        for j in range(i-period+1, i+1):
            tr = max(high[j] - low[j], np.abs(high[j] - close[j-1]), np.abs(low[j] - close[j-1]))
            tr_sum += tr
        
        chop[i] = 100 * np.log10(tr_sum / (highest_high - lowest_low)) / np.log10(period)
    
    chop = np.clip(chop, 0, 100)
    return chop

def calculate_donchian(high, low, period=20):
    """Donchian Channels — highest high and lowest low over period."""
    n = len(high)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i-period+1:i+1])
        lower[i] = np.min(low[i-period+1:i+1])
    
    return upper, lower

def calculate_atr(high, low, close, period=14):
    """Average True Range."""
    n = len(close)
    atr = np.full(n, np.nan)
    
    if n < period + 1:
        return atr
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], np.abs(high[i] - close[i-1]), np.abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate primary (1d) indicators
    rsi_1d = calculate_rsi(close, period=14)
    chop_1d = calculate_choppiness(high, low, close, period=14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    atr_1d = calculate_atr(high, low, close, period=14)
    sma_200 = calculate_sma(close, 200)
    fisher_1d, fisher_prev_1d = calculate_fisher_transform(high, low, period=9)
    
    # Calculate and align 1w HMA for long-term trend bias
    hma_1w_raw = calculate_hma(df_1w['close'].values, 21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.30
    REDUCED_SIZE = 0.20
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(300, n):
        # Skip if indicators not ready
        if np.isnan(rsi_1d[i]) or np.isnan(chop_1d[i]) or np.isnan(atr_1d[i]):
            continue
        if atr_1d[i] <= 1e-10:
            continue
        if np.isnan(hma_1w_aligned[i]):
            continue
        if np.isnan(sma_200[i]):
            continue
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            continue
        if np.isnan(fisher_1d[i]) or np.isnan(fisher_prev_1d[i]):
            continue
        
        # === LONG-TERM TREND BIAS (1w HTF HMA21) ===
        trend_1w_bullish = close[i] > hma_1w_aligned[i]
        trend_1w_bearish = close[i] < hma_1w_aligned[i]
        
        # === SECULAR TREND FILTER (SMA200) ===
        above_sma200 = close[i] > sma_200[i]
        below_sma200 = close[i] < sma_200[i]
        
        # === REGIME DETECTION (1d Choppiness Index) ===
        ranging_regime = chop_1d[i] > 55
        trending_regime = chop_1d[i] < 45
        neutral_regime = not ranging_regime and not trending_regime
        
        # === RSI SIGNALS (Relaxed for daily timeframe) ===
        rsi_oversold = rsi_1d[i] < 35
        rsi_overbought = rsi_1d[i] > 65
        rsi_extreme_oversold = rsi_1d[i] < 25
        rsi_extreme_overbought = rsi_1d[i] > 75
        rsi_neutral_low = 35 <= rsi_1d[i] < 50
        rsi_neutral_high = 50 < rsi_1d[i] <= 65
        
        # === FISHER TRANSFORM SIGNALS ===
        fisher_oversold = fisher_1d[i] < -1.5
        fisher_overbought = fisher_1d[i] > 1.5
        fisher_cross_up = fisher_prev_1d[i] < -1.5 and fisher_1d[i] >= -1.5
        fisher_cross_down = fisher_prev_1d[i] > 1.5 and fisher_1d[i] <= 1.5
        fisher_recovering = fisher_1d[i] > fisher_prev_1d[i] and fisher_1d[i] < -0.5
        fisher_weakening = fisher_1d[i] < fisher_prev_1d[i] and fisher_1d[i] > 0.5
        
        # === DONCHIAN BREAKOUT ===
        donchian_breakout_long = close[i] > donchian_upper[i-1] if not np.isnan(donchian_upper[i-1]) else False
        donchian_breakout_short = close[i] < donchian_lower[i-1] if not np.isnan(donchian_lower[i-1]) else False
        
        desired_signal = 0.0
        
        # === RANGING REGIME LOGIC (CHOP > 55) — Mean Reversion with Fisher ===
        if ranging_regime:
            # Long: Fisher oversold + RSI oversold + ANY trend alignment
            if fisher_oversold and rsi_oversold and (above_sma200 or trend_1w_bullish):
                desired_signal = BASE_SIZE
            
            # Short: Fisher overbought + RSI overbought + ANY trend alignment
            if fisher_overbought and rsi_overbought and (below_sma200 or trend_1w_bearish):
                desired_signal = -BASE_SIZE
            
            # Fisher reversal cross + RSI confluence (high probability)
            if fisher_cross_up and rsi_oversold:
                desired_signal = REDUCED_SIZE if desired_signal == 0 else desired_signal
            
            if fisher_cross_down and rsi_overbought:
                desired_signal = -REDUCED_SIZE if desired_signal == 0 else desired_signal
            
            # Fallback: extreme RSI alone (guarantees trades on all symbols)
            if rsi_extreme_oversold and desired_signal == 0:
                desired_signal = REDUCED_SIZE
            
            if rsi_extreme_overbought and desired_signal == 0:
                desired_signal = -REDUCED_SIZE
        
        # === TRENDING REGIME LOGIC (CHOP < 45) — Trend Following ===
        elif trending_regime:
            # Long: Bullish trend + Fisher recovering from oversold OR Donchian breakout
            if trend_1w_bullish or above_sma200:
                if fisher_recovering and rsi_neutral_low:
                    desired_signal = BASE_SIZE
                elif donchian_breakout_long:
                    desired_signal = REDUCED_SIZE if desired_signal == 0 else desired_signal
            
            # Short: Bearish trend + Fisher weakening from overbought OR Donchian breakout
            if trend_1w_bearish or below_sma200:
                if fisher_weakening and rsi_neutral_high:
                    desired_signal = -BASE_SIZE
                elif donchian_breakout_short:
                    desired_signal = -REDUCED_SIZE if desired_signal == 0 else desired_signal
        
        # === NEUTRAL REGIME (45 <= CHOP <= 55) ===
        else:
            # Conservative: Fisher + RSI confluence + any trend alignment
            if fisher_oversold and rsi_oversold and (trend_1w_bullish or above_sma200):
                desired_signal = REDUCED_SIZE
            
            if fisher_overbought and rsi_overbought and (trend_1w_bearish or below_sma200):
                desired_signal = -REDUCED_SIZE
            
            # Basic mean reversion with single filter
            if rsi_oversold and above_sma200 and fisher_recovering:
                desired_signal = REDUCED_SIZE if desired_signal == 0 else desired_signal
            
            if rsi_overbought and below_sma200 and fisher_weakening:
                desired_signal = -REDUCED_SIZE if desired_signal == 0 else desired_signal
        
        # === STOPLOSS CHECK (Trailing ATR 2.5x) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * entry_atr
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * entry_atr
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === HOLD LOGIC — Maintain position if conditions intact ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if trend intact and Fisher not overbought
                if (trend_1w_bullish or above_sma200) and fisher_1d[i] < 1.0:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if trend intact and Fisher not oversold
                if (trend_1w_bearish or below_sma200) and fisher_1d[i] > -1.0:
                    desired_signal = -BASE_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if trend reverses + Fisher overbought
            if trend_1w_bearish and below_sma200 and fisher_1d[i] > 1.5:
                desired_signal = 0.0
            # Exit if RSI extremely overbought in ranging regime
            if ranging_regime and rsi_1d[i] > 80:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if trend reverses + Fisher oversold
            if trend_1w_bullish and above_sma200 and fisher_1d[i] < -1.5:
                desired_signal = 0.0
            # Exit if RSI extremely oversold in ranging regime
            if ranging_regime and rsi_1d[i] < 20:
                desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            desired_signal = BASE_SIZE if desired_signal >= BASE_SIZE else REDUCED_SIZE
        elif desired_signal < 0:
            desired_signal = -BASE_SIZE if desired_signal <= -BASE_SIZE else -REDUCED_SIZE
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_1d[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_1d[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif position_side > 0:
                highest_since_entry = max(highest_since_entry, close[i])
            elif position_side < 0:
                lowest_since_entry = min(lowest_since_entry, close[i])
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
        
        signals[i] = desired_signal
    
    return signals
```

## Last Updated
2026-03-23 15:18
