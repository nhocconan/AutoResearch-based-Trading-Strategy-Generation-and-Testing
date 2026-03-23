# Strategy: mtf_1d_hma_donchian_rsi_pullback_1w_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.090 | +24.1% | -9.5% | 63 | PASS |
| ETHUSDT | 0.059 | +22.3% | -15.6% | 70 | PASS |
| SOLUSDT | 0.662 | +80.2% | -26.1% | 56 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.068 | -4.4% | -9.3% | 46 | FAIL |
| ETHUSDT | -1.457 | -16.6% | -19.9% | 43 | FAIL |
| SOLUSDT | 0.020 | +5.2% | -12.2% | 20 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
Experiment #543: 1d Primary + 1w HTF — HMA Trend + Donchian Breakout + RSI Pullback

Hypothesis: After analyzing 480+ failed strategies, the key insight is:
- 1d timeframe with 1w HTF filter worked well (#537 Sharpe=0.397, #536 Sharpe=0.159)
- Complex regime switching (Choppiness/Fisher) consistently failed
- Simple trend-following with pullback entries generates consistent trades
- Donchian breakouts capture momentum moves in crypto's trending nature
- RSI pullback entries avoid chasing breakouts at tops

This strategy combines:
1. 1w HMA(21) for major weekly trend direction (HTF filter)
2. 1d HMA(16/48) crossover for medium-term trend confirmation
3. Donchian(20) breakout for momentum entry trigger
4. RSI(14) pullback filter (35-55 for long, 45-65 for short)
5. ATR(14) 2.5x trailing stop for risk management
6. ADX(14) > 18 filter to avoid choppy whipsaws

Why this might work:
- 1w trend filter prevents counter-trend trades (major failure mode in 2022 crash)
- Donchian breakout captures sustained momentum moves
- RSI pullback ensures we're not buying tops/selling bottoms
- 1d TF targets 25-40 trades/year (optimal fee/trade ratio)
- Simple logic = consistent signals across BTC/ETH/SOL

Position sizing: 0.28 (discrete, max 0.40 per rules)
Stoploss: 2.5 * ATR trailing (signal → 0 when hit)
Target: >=30 trades/symbol on train, >=3 on test, Sharpe > 0 all symbols
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_hma_donchian_rsi_pullback_1w_v1"
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
    """Calculate RSI using Wilder's smoothing."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi.values

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index) for trend strength."""
    n = len(close)
    
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    
    plus_dm = high_s.diff()
    minus_dm = -low_s.diff()
    
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    
    mask = (plus_dm > 0) & (minus_dm > 0)
    plus_dm_vals = plus_dm.values.copy()
    minus_dm_vals = minus_dm.values.copy()
    plus_dm_vals[mask] = np.where(plus_dm_vals[mask] > minus_dm_vals[mask], plus_dm_vals[mask], 0)
    minus_dm_vals[mask] = np.where(minus_dm_vals[mask] > plus_dm_vals[mask], minus_dm_vals[mask], 0)
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    plus_dm_s = pd.Series(plus_dm_vals).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_s = pd.Series(minus_dm_vals).ewm(span=period, min_periods=period, adjust=False).mean().values
    tr_s = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    plus_di = 100.0 * plus_dm_s / (tr_s + 1e-10)
    minus_di = 100.0 * minus_dm_s / (tr_s + 1e-10)
    
    dx = 100.0 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average (HMA) - reduces lag vs EMA."""
    n = period
    half = n // 2
    sqrt_n = int(np.sqrt(n))
    
    close_s = pd.Series(close)
    
    def wma(series, span):
        weights = np.arange(1, span + 1)
        return series.rolling(window=span, min_periods=span).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )
    
    wma_half = wma(close_s, half)
    wma_full = wma(close_s, n)
    hma_raw = 2.0 * wma_half - wma_full
    hma = wma(hma_raw, sqrt_n)
    
    return hma.values

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (upper/lower bands)."""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    mid = (upper + lower) / 2.0
    return upper, lower, mid

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load 1w HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w HTF HMA for major trend direction
    hma_1w_21 = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_50 = calculate_hma(df_1w['close'].values, period=50)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1w_21_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_21)
    hma_1w_50_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_50)
    
    # Calculate 1d indicators
    atr_14 = calculate_atr(high, low, close, 14)
    adx_14 = calculate_adx(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    
    # 1d HMA for trend confirmation
    hma_1d_16 = calculate_hma(close, period=16)
    hma_1d_48 = calculate_hma(close, period=48)
    
    # Donchian channels for breakout detection
    donchian_upper, donchian_lower, donchian_mid = calculate_donchian(high, low, 20)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    POSITION_SIZE = 0.28
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Track previous values for crossover detection
    prev_hma_16 = np.zeros(n)
    prev_hma_16[1:] = hma_1d_16[:-1]
    prev_hma_48 = np.zeros(n)
    prev_hma_48[1:] = hma_1d_48[:-1]
    
    # Track Donchian breakout
    prev_donchian_upper = np.zeros(n)
    prev_donchian_upper[1:] = donchian_upper[:-1]
    prev_donchian_lower = np.zeros(n)
    prev_donchian_lower[1:] = donchian_lower[:-1]
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            signals[i] = signals[i-1] if i > 0 else 0.0
            continue
        if np.isnan(hma_1w_21_aligned[i]) or np.isnan(hma_1w_50_aligned[i]):
            signals[i] = signals[i-1] if i > 0 else 0.0
            continue
        if np.isnan(hma_1d_16[i]) or np.isnan(hma_1d_48[i]):
            signals[i] = signals[i-1] if i > 0 else 0.0
            continue
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            signals[i] = signals[i-1] if i > 0 else 0.0
            continue
        if np.isnan(adx_14[i]) or np.isnan(rsi_14[i]):
            signals[i] = signals[i-1] if i > 0 else 0.0
            continue
        
        # === 1W MAJOR TREND (primary direction filter) ===
        bull_regime_1w = close[i] > hma_1w_21_aligned[i]
        bear_regime_1w = close[i] < hma_1w_21_aligned[i]
        
        # 1w HMA slope for trend strength
        hma_1w_slope_bull = hma_1w_21_aligned[i] > hma_1w_50_aligned[i]
        hma_1w_slope_bear = hma_1w_21_aligned[i] < hma_1w_50_aligned[i]
        
        # === 1D TREND CONFIRMATION ===
        # HMA crossover (fast above slow = bull)
        hma_bull_cross = hma_1d_16[i] > hma_1d_48[i]
        hma_bear_cross = hma_1d_16[i] < hma_1d_48[i]
        
        # HMA crossover confirmation (just crossed)
        hma_bull_crossed = hma_bull_cross and (prev_hma_16[i] <= prev_hma_48[i])
        hma_bear_crossed = hma_bear_cross and (prev_hma_16[i] >= prev_hma_48[i])
        
        # === DONCHIAN BREAKOUT ===
        donchian_breakout_long = close[i] > prev_donchian_upper[i]
        donchian_breakout_short = close[i] < prev_donchian_lower[i]
        
        # === ADX FILTER (trending market) ===
        trending = adx_14[i] > 18.0  # Market is trending (lower threshold for more trades)
        
        # === RSI PULLBACK FILTER ===
        rsi_pullback_long = 35.0 < rsi_14[i] < 55.0  # Pullback in uptrend
        rsi_pullback_short = 45.0 < rsi_14[i] < 65.0  # Pullback in downtrend
        rsi_neutral_long = rsi_14[i] < 60.0  # Not overbought for long
        rsi_neutral_short = rsi_14[i] > 40.0  # Not oversold for short
        
        # === ENTRY LOGIC — MULTIPLE CONDITIONS FOR TRADE FREQUENCY ===
        new_signal = 0.0
        
        # LONG ENTRIES (OR logic for frequency - key lesson from failures)
        # Condition 1: 1w bull + 1d HMA bull + Donchian breakout + RSI ok
        if bull_regime_1w and hma_bull_cross and donchian_breakout_long and rsi_neutral_long:
            new_signal = POSITION_SIZE
        # Condition 2: 1w bull + 1d HMA crossed bull + trending + RSI pullback
        elif bull_regime_1w and hma_bull_crossed and trending and rsi_pullback_long:
            new_signal = POSITION_SIZE
        # Condition 3: 1w bull + HMA slope bull + Donchian breakout
        elif bull_regime_1w and hma_1w_slope_bull and donchian_breakout_long:
            new_signal = POSITION_SIZE * 0.9
        # Condition 4: 1w bull + 1d HMA bull + RSI pullback (no breakout needed)
        elif bull_regime_1w and hma_bull_cross and rsi_pullback_long:
            new_signal = POSITION_SIZE * 0.8
        # Condition 5: Strong 1w bull (slope) + 1d HMA bull + trending
        elif bull_regime_1w and hma_1w_slope_bull and hma_bull_cross and trending:
            new_signal = POSITION_SIZE * 0.9
        
        # SHORT ENTRIES (mirror logic)
        if new_signal == 0.0:
            # Condition 1: 1w bear + 1d HMA bear + Donchian breakout + RSI ok
            if bear_regime_1w and hma_bear_cross and donchian_breakout_short and rsi_neutral_short:
                new_signal = -POSITION_SIZE
            # Condition 2: 1w bear + 1d HMA crossed bear + trending + RSI pullback
            elif bear_regime_1w and hma_bear_crossed and trending and rsi_pullback_short:
                new_signal = -POSITION_SIZE
            # Condition 3: 1w bear + HMA slope bear + Donchian breakout
            elif bear_regime_1w and hma_1w_slope_bear and donchian_breakout_short:
                new_signal = -POSITION_SIZE * 0.9
            # Condition 4: 1w bear + 1d HMA bear + RSI pullback (no breakout needed)
            elif bear_regime_1w and hma_bear_cross and rsi_pullback_short:
                new_signal = -POSITION_SIZE * 0.8
            # Condition 5: Strong 1w bear (slope) + 1d HMA bear + trending
            elif bear_regime_1w and hma_1w_slope_bear and hma_bear_cross and trending:
                new_signal = -POSITION_SIZE * 0.9
        
        # === HOLD POSITION LOGIC ===
        # If already in position, maintain unless exit conditions hit
        if in_position and new_signal == 0.0:
            new_signal = signals[i-1] if i > 0 else 0.0
        
        # === STOPLOSS CHECK (2.5 * ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * atr_14[i]
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            if lowest_since_entry == 0.0:
                lowest_since_entry = close[i]
            else:
                lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * atr_14[i]
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            new_signal = 0.0
        
        # === EXIT CONDITIONS (regime flip or weak trend) ===
        # Exit long on 1w regime flip to bear
        if in_position and position_side > 0:
            if bear_regime_1w and hma_1w_slope_bear:
                new_signal = 0.0
            elif adx_14[i] < 15.0:  # Trend weakening significantly
                new_signal = 0.0
        
        # Exit short on 1w regime flip to bull
        if in_position and position_side < 0:
            if bull_regime_1w and hma_1w_slope_bull:
                new_signal = 0.0
            elif adx_14[i] < 15.0:  # Trend weakening significantly
                new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Flip position
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = new_signal
    
    return signals
```

## Last Updated
2026-03-23 05:47
