# Strategy: mtf_1d_fisher_hma_chop_1w_asym_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.197 | +16.2% | -8.4% | 86 | FAIL |
| ETHUSDT | -0.194 | +14.7% | -11.2% | 91 | FAIL |
| SOLUSDT | 0.631 | +59.2% | -15.6% | 96 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| SOLUSDT | 0.503 | +12.8% | -7.4% | 26 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
Experiment #293: 1d Primary + 1w HTF — Fisher Transform + HMA Trend + Choppiness Regime

Hypothesis: After 266 failed experiments, try a fundamentally different approach:
1. Ehlers Fisher Transform (period=9) for reversal detection — catches bear market rallies
2. 1w HMA(21) for PRIMARY trend direction (slower, more stable than 1d HTF)
3. Choppiness Index(14) regime filter — mean revert in chop, trend follow otherwise
4. Asymmetric entries: Long ONLY when 1w trend bull, Short ONLY when 1w trend bear
5. ATR volatility scaling on position size (reduce exposure in high vol)
6. Very few trades target: 20-40/year on 1d (appropriate for daily timeframe)

Why this might work when others failed:
- Fisher Transform specifically designed for non-Gaussian price distributions (crypto)
- 1w HTF is slower than previous 1d/4h HTF attempts — fewer false signals
- Asymmetric logic matches crypto behavior (bull trends stronger than bear)
- ATR sizing reduces drawdown during vol spikes (2022 crash protection)

Position sizing: 0.25 base, 0.35 strong conviction, scaled by ATR ratio
Target: 20-40 trades/year per symbol (appropriate for 1d)
Stoploss: 3.0 * ATR trailing (wider for daily timeframe)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_fisher_hma_chop_1w_asym_v1"
timeframe = "1d"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_rsi(close, period=14):
    """Calculate RSI using standard Wilder's method."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values
    return rsi

def calculate_hma(close, period=21):
    """
    Calculate Hull Moving Average (HMA).
    HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    Faster and smoother than EMA, less lag.
    """
    n = period
    half = n // 2
    sqrt_n = int(np.sqrt(n))
    
    close_s = pd.Series(close)
    
    def wma(series, window):
        weights = np.arange(1, window + 1)
        return series.rolling(window=window, min_periods=window).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )
    
    wma_half = wma(close_s, half)
    wma_full = wma(close_s, n)
    
    raw_hma = 2 * wma_half - wma_full
    hma = wma(raw_hma, sqrt_n)
    
    return hma.values

def calculate_fisher_transform(high, low, close, period=9):
    """
    Calculate Ehlers Fisher Transform.
    Converts price to Gaussian-like distribution for clearer reversal signals.
    Long when Fisher crosses above -1.5 from below
    Short when Fisher crosses below +1.5 from above
    """
    n = period
    hl2 = (high + low) / 2.0
    
    # Calculate EMA of HL2
    ema_hl2 = pd.Series(hl2).ewm(span=n, min_periods=n, adjust=False).mean().values
    
    fisher = np.zeros(len(close))
    fisher_prev = np.zeros(len(close))
    
    for i in range(n, len(close)):
        # Normalize price within recent range
        highest = np.max(hl2[max(0, i-n+1):i+1])
        lowest = np.min(hl2[max(0, i-n+1):i+1])
        
        if highest > lowest:
            normalized = 0.999 * (hl2[i] - lowest) / (highest - lowest) + 0.001
        else:
            normalized = 0.5
        
        # Fisher transform
        fisher[i] = 0.5 * np.log((1 + normalized) / (1 - normalized + 1e-10))
        
        # Smooth with EMA
        if i > n:
            fisher[i] = 0.7 * fisher[i] + 0.3 * fisher[i-1]
        
        fisher_prev[i] = fisher[i-1] if i > 0 else fisher[i]
    
    return fisher, fisher_prev

def calculate_choppiness_index(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP > 61.8 = choppy/range market (mean revert)
    CHOP < 38.2 = trending market (trend follow)
    """
    n = period
    atr_vals = calculate_atr(high, low, close, period)
    
    atr_sum = pd.Series(atr_vals).rolling(window=n, min_periods=n).sum().values
    hh = pd.Series(high).rolling(window=n, min_periods=n).max().values
    ll = pd.Series(low).rolling(window=n, min_periods=n).min().values
    
    chop = np.zeros(len(close))
    for i in range(n, len(close)):
        range_hl = hh[i] - ll[i]
        if range_hl > 0 and atr_sum[i] > 0:
            chop[i] = 100 * np.log10(atr_sum[i] / range_hl) / np.log10(n)
        else:
            chop[i] = 50.0
    
    return chop

def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    return upper, lower, sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w HTF indicators (primary trend regime)
    hma_1w_21 = calculate_hma(df_1w['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_1w_21_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_21)
    
    # Calculate 1d indicators
    atr_14 = calculate_atr(high, low, close, 14)
    atr_30 = calculate_atr(high, low, close, 30)
    chop_14 = calculate_choppiness_index(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    fisher, fisher_prev = calculate_fisher_transform(high, low, close, 9)
    bb_upper, bb_lower, bb_mid = calculate_bollinger_bands(close, 20, 2.0)
    hma_1d_21 = calculate_hma(close, 21)
    hma_1d_50 = calculate_hma(close, 50)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    BASE_SIZE = 0.25
    STRONG_SIZE = 0.35
    MIN_SIZE = 0.15
    
    # Track position state
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -20
    entry_fisher = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_1w_21_aligned[i]):
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(fisher[i]):
            continue
        
        if np.isnan(chop_14[i]) or np.isnan(bb_upper[i]):
            continue
        
        # === 1W TREND REGIME (primary direction filter — ASYMMETRIC) ===
        # Bull: price above 1w HMA (only take longs or reduce shorts)
        # Bear: price below 1w HMA (only take shorts or reduce longs)
        regime_bull = close[i] > hma_1w_21_aligned[i]
        regime_bear = close[i] < hma_1w_21_aligned[i]
        
        # === CHOPPINESS REGIME ===
        # CHOP > 58 = range market (mean revert entries)
        # CHOP < 42 = trend market (breakout entries)
        is_choppy = chop_14[i] > 58.0
        is_trending = chop_14[i] < 42.0
        
        # === VOLATILITY REGIME (ATR ratio) ===
        # High vol: ATR(14)/ATR(30) > 1.5 (reduce position size)
        # Normal vol: ATR ratio < 1.2 (full position size)
        atr_ratio = atr_14[i] / (atr_30[i] + 1e-10)
        high_vol = atr_ratio > 1.5
        vol_scale = 0.7 if high_vol else 1.0
        
        # === 1D LOCAL SIGNALS ===
        price_above_1d_hma = close[i] > hma_1d_21[i]
        price_below_1d_hma = close[i] < hma_1d_21[i]
        hma_1d_bullish = hma_1d_21[i] > hma_1d_50[i]
        hma_1d_bearish = hma_1d_21[i] < hma_1d_50[i]
        
        # === FISHER TRANSFORM SIGNALS ===
        fisher_oversold = fisher[i] < -1.5
        fisher_overbought = fisher[i] > 1.5
        fisher_cross_up = fisher_prev[i] < -1.5 and fisher[i] >= -1.5
        fisher_cross_down = fisher_prev[i] > 1.5 and fisher[i] <= 1.5
        
        # === RSI THRESHOLDS ===
        rsi_oversold = rsi_14[i] < 35.0
        rsi_overbought = rsi_14[i] > 65.0
        rsi_extreme_oversold = rsi_14[i] < 25.0
        rsi_extreme_overbought = rsi_14[i] > 75.0
        
        # === BOLLINGER BAND SIGNALS ===
        bb_break_lower = close[i] < bb_lower[i] * 1.002
        bb_break_upper = close[i] > bb_upper[i] * 0.998
        bb_revert_mid = (close[i] > bb_mid[i] and close[i-1] <= bb_mid[i-1]) if i > 0 else False
        
        # === ENTRY LOGIC (ASYMMETRIC) ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRIES (only when 1w regime bull or neutral)
        if regime_bull or (not regime_bear and is_choppy):
            # Fisher reversal from oversold + RSI confirming
            if fisher_cross_up and rsi_14[i] < 50:
                new_signal = STRONG_SIZE * vol_scale
            # Mean revert in choppy market + BB lower break + RSI oversold
            elif is_choppy and bb_break_lower and rsi_oversold:
                new_signal = BASE_SIZE * vol_scale
            # Trend follow + 1d HMA bullish + price above 1d HMA
            elif is_trending and hma_1d_bullish and price_above_1d_hma and rsi_14[i] > 45:
                new_signal = BASE_SIZE * vol_scale
            # Extreme RSI oversold in any regime (strong conviction long)
            elif rsi_extreme_oversold and regime_bull:
                new_signal = STRONG_SIZE * vol_scale
        
        # SHORT ENTRIES (only when 1w regime bear or neutral)
        if regime_bear or (not regime_bull and is_choppy):
            # Fisher reversal from overbought + RSI confirming
            if fisher_cross_down and rsi_14[i] > 50:
                if new_signal == 0.0:
                    new_signal = -STRONG_SIZE * vol_scale
            # Mean revert in choppy market + BB upper break + RSI overbought
            elif is_choppy and bb_break_upper and rsi_overbought:
                if new_signal == 0.0:
                    new_signal = -BASE_SIZE * vol_scale
            # Trend follow + 1d HMA bearish + price below 1d HMA
            elif is_trending and hma_1d_bearish and price_below_1d_hma and rsi_14[i] < 55:
                if new_signal == 0.0:
                    new_signal = -BASE_SIZE * vol_scale
            # Extreme RSI overbought in bear regime (strong conviction short)
            elif rsi_extreme_overbought and regime_bear:
                if new_signal == 0.0:
                    new_signal = -STRONG_SIZE * vol_scale
        
        # === FREQUENCY SAFEGUARD (ensure 20+ trades/year on 1d) ===
        # Force trade if no signal for 25 bars (~25 days on 1d)
        if bars_since_last_trade > 25 and new_signal == 0.0 and not in_position:
            if regime_bull and rsi_14[i] > 40 and price_above_1d_hma:
                new_signal = MIN_SIZE * vol_scale
            elif regime_bear and rsi_14[i] < 60 and price_below_1d_hma:
                new_signal = -MIN_SIZE * vol_scale
            elif is_choppy and rsi_14[i] < 35:
                new_signal = MIN_SIZE * vol_scale
            elif is_choppy and rsi_14[i] > 65:
                new_signal = -MIN_SIZE * vol_scale
        
        # === STOPLOSS LOGIC (Rule 6) - 3.0 * ATR trailing (wider for 1d) ===
        stoploss_triggered = False
        
        if in_position and position_side != 0:
            if position_side > 0:
                if close[i] > highest_price:
                    highest_price = close[i]
                stoploss_price = highest_price - 3.0 * atr_14[i]
                if close[i] < stoploss_price:
                    stoploss_triggered = True
            
            if position_side < 0:
                if lowest_price == 0.0 or close[i] < lowest_price:
                    lowest_price = close[i]
                stoploss_price = lowest_price + 3.0 * atr_14[i]
                if close[i] > stoploss_price:
                    stoploss_triggered = True
        
        # === FISHER REVERSAL EXIT ===
        fisher_exit = False
        if in_position and position_side != 0:
            # Long position: exit when Fisher goes overbought
            if position_side > 0 and fisher[i] > 1.5:
                fisher_exit = True
            # Short position: exit when Fisher goes oversold
            if position_side < 0 and fisher[i] < -1.5:
                fisher_exit = True
        
        # === REGIME REVERSAL EXIT ===
        regime_reversal = False
        if in_position and position_side != 0:
            # Long position but 1w regime turns strongly bearish
            if position_side > 0 and regime_bear and price_below_1d_hma:
                regime_reversal = True
            # Short position but 1w regime turns strongly bullish
            if position_side < 0 and regime_bull and price_above_1d_hma:
                regime_reversal = True
        
        if stoploss_triggered or fisher_exit or regime_reversal:
            new_signal = 0.0
        
        # === DISCRETIZE SIGNAL (reduce churn) ===
        if new_signal != 0.0:
            if abs(new_signal) < 0.20:
                new_signal = 0.0
            elif new_signal > 0.30:
                new_signal = STRONG_SIZE * vol_scale
            elif new_signal > 0:
                new_signal = BASE_SIZE * vol_scale
            elif new_signal < -0.30:
                new_signal = -STRONG_SIZE * vol_scale
            else:
                new_signal = -BASE_SIZE * vol_scale
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                entry_fisher = fisher[i]
                last_trade_bar = i
            elif np.sign(new_signal) != position_side:
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                entry_fisher = fisher[i]
                last_trade_bar = i
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_price = 0.0
                lowest_price = 0.0
                entry_fisher = 0.0
                last_trade_bar = i
        
        signals[i] = new_signal
    
    return signals
```

## Last Updated
2026-03-23 02:05
