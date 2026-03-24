# Strategy: mtf_1d_dual_regime_crsi_chop_1w_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.003 | +20.2% | -17.5% | 68 | PASS |
| ETHUSDT | -0.454 | -5.9% | -18.2% | 87 | FAIL |
| SOLUSDT | -0.944 | -36.4% | -41.5% | 80 | FAIL |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.372 | +9.8% | -9.0% | 26 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
Experiment #103: 1d Primary + 1w HTF — Dual Regime Adaptive Strategy

Hypothesis: Previous 1d strategies failed because they used single-regime logic 
that doesn't adapt to market conditions. This strategy switches between:
- TREND MODE (CHOP < 50): Follow 1w HMA direction with 1d pullback entries
- RANGE MODE (CHOP > 50): Mean revert at Bollinger Band extremes with RSI filter

Key innovations:
1) 1w HMA slope determines macro bias (proven in #079)
2) Choppiness Index (14) switches between trend/range logic
3) Connors RSI for entry timing (RSI2 + RSI_Streak + PercentRank)
4) Bollinger Band squeeze detection for volatility expansion
5) Conservative sizing: 0.25 base, 0.30 max with confluence
6) ATR(14) trailing stop at 2.5x

Why this should work on 1d:
- Natural trade frequency: 20-40 trades/year (low fee drag)
- Regime adaptation works in both 2021-2024 bull and 2025 bear
- 1w HTF prevents counter-trend trades in bear markets
- CRSI proven 75% win rate on daily timeframes
- Simple conditions = trades actually generate on all symbols

Position size: 0.25 base, 0.30 max
Stoploss: 2.5*ATR trailing
Target: 25-40 trades/year, Sharpe > 0.5 on ALL symbols
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_dual_regime_crsi_chop_1w_v1"
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
    """Calculate RSI."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi = rsi.fillna(50.0).values
    return rsi

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average."""
    close_s = pd.Series(close)
    wma1 = close_s.ewm(span=period//2, min_periods=period//2, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma_diff = 2.0 * wma1 - wma2
    hma = wma_diff.ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean()
    return hma.values

def calculate_bollinger_bands(close, period=20, std_dev=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    upper = sma + (std_dev * std)
    lower = sma - (std_dev * std)
    return sma.values, upper.values, lower.values

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP > 61.8 = ranging market
    CHOP < 38.2 = trending market
    """
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr_sum = pd.Series(tr).rolling(window=period, min_periods=period).sum().values
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    price_range = highest_high - lowest_low
    chop = 100.0 * np.log10(atr_sum / (price_range + 1e-10)) / np.log10(period)
    chop = np.clip(chop, 0, 100)
    chop = np.nan_to_num(chop, nan=50.0)
    return chop

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(close, 3) + RSI(Streak, 2) + PercentRank(100)) / 3
    """
    close_s = pd.Series(close)
    
    # RSI on close (short period)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean()
    avg_loss = loss.ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_close = 100.0 - (100.0 / (1.0 + rs))
    rsi_close = rsi_close.fillna(50.0).values
    
    # RSI on streak
    streak = np.zeros(len(close))
    for i in range(1, len(close)):
        if delta.iloc[i] > 0:
            streak[i] = streak[i-1] + 1
        elif delta.iloc[i] < 0:
            streak[i] = streak[i-1] - 1
        else:
            streak[i] = 0
    
    streak_s = pd.Series(streak)
    streak_gain = streak_s.diff().where(streak_s.diff() > 0, 0.0)
    streak_loss = -streak_s.diff().where(streak_s.diff() < 0, 0.0)
    avg_streak_gain = streak_gain.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    avg_streak_loss = streak_loss.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    rs_streak = avg_streak_gain / (avg_streak_loss + 1e-10)
    rsi_streak = 100.0 - (100.0 / (1.0 + rs_streak))
    rsi_streak = rsi_streak.fillna(50.0).values
    
    # Percent rank
    percent_rank = pd.Series(close).rolling(window=rank_period, min_periods=rank_period).apply(
        lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min() + 1e-10) * 100,
        raw=False
    ).values
    percent_rank = np.nan_to_num(percent_rank, nan=50.0)
    
    crsi = (rsi_close + rsi_streak + percent_rank) / 3.0
    crsi = np.nan_to_num(crsi, nan=50.0)
    return crsi

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w HMA for macro trend
    hma_1w = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 1w HMA slope (trend strength)
    hma_1w_slope = np.zeros(n)
    for i in range(1, n):
        if not np.isnan(hma_1w_aligned[i]) and not np.isnan(hma_1w_aligned[i-1]) and hma_1w_aligned[i-1] != 0:
            hma_1w_slope[i] = (hma_1w_aligned[i] - hma_1w_aligned[i-1]) / hma_1w_aligned[i-1] * 100
        else:
            hma_1w_slope[i] = 0.0
    
    # Calculate 1d indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, period=14)
    crsi = calculate_connors_rsi(close)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    bb_mid, bb_upper, bb_lower = calculate_bollinger_bands(close, period=20, std_dev=2.0)
    hma_1d_16 = calculate_hma(close, period=16)
    hma_1d_48 = calculate_hma(close, period=48)
    
    signals = np.zeros(n)
    POSITION_SIZE_BASE = 0.25
    POSITION_SIZE_MAX = 0.30
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_price = 0.0
    
    for i in range(250, n):
        # Skip if indicators not ready
        if np.isnan(hma_1w_aligned[i]) or np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        if np.isnan(rsi_14[i]) or np.isnan(crsi[i]) or np.isnan(chop_14[i]):
            continue
        if np.isnan(bb_mid[i]) or np.isnan(hma_1d_16[i]) or np.isnan(hma_1d_48[i]):
            continue
        
        # === HTF TREND BIAS (1w HMA) ===
        price_above_hma_1w = close[i] > hma_1w_aligned[i]
        price_below_hma_1w = close[i] < hma_1w_aligned[i]
        hma_slope_positive = hma_1w_slope[i] > 0.2
        hma_slope_negative = hma_1w_slope[i] < -0.2
        
        # === CHOPPINESS REGIME ===
        chop_trending = chop_14[i] < 50.0  # trending market
        chop_ranging = chop_14[i] >= 50.0  # ranging market
        
        # === 1d HMA CROSSOVER ===
        hma_bullish = hma_1d_16[i] > hma_1d_48[i]
        hma_bearish = hma_1d_16[i] < hma_1d_48[i]
        
        # === BOLLINGER BAND POSITION ===
        bb_range = bb_upper[i] - bb_lower[i] + 1e-10
        bb_pct = (close[i] - bb_lower[i]) / bb_range
        near_bb_lower = bb_pct < 0.20  # near lower band
        near_bb_upper = bb_pct > 0.80  # near upper band
        
        # === CONNORS RSI SIGNALS ===
        crsi_oversold = crsi[i] < 20.0  # extreme oversold
        crsi_overbought = crsi[i] > 80.0  # extreme overbought
        rsi_oversold = rsi_14[i] < 40.0
        rsi_overbought = rsi_14[i] > 60.0
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # --- LONG ENTRY ---
        if chop_trending:
            # TREND MODE: Follow weekly trend with daily pullback
            if price_above_hma_1w and hma_bullish:
                if rsi_oversold or crsi_oversold or near_bb_lower:
                    new_signal = POSITION_SIZE_BASE
                    if (rsi_oversold and crsi_oversold) or (near_bb_lower and crsi_oversold):
                        new_signal = POSITION_SIZE_MAX
        else:
            # RANGE MODE: Mean revert at extremes
            if near_bb_lower and (rsi_oversold or crsi_oversold):
                new_signal = POSITION_SIZE_BASE
                if crsi_oversold:
                    new_signal = POSITION_SIZE_MAX
        
        # --- SHORT ENTRY ---
        if chop_trending:
            # TREND MODE: Follow weekly trend with daily pullback
            if price_below_hma_1w and hma_bearish:
                if rsi_overbought or crsi_overbought or near_bb_upper:
                    new_signal = -POSITION_SIZE_BASE
                    if (rsi_overbought and crsi_overbought) or (near_bb_upper and crsi_overbought):
                        new_signal = -POSITION_SIZE_MAX
        else:
            # RANGE MODE: Mean revert at extremes
            if near_bb_upper and (rsi_overbought or crsi_overbought):
                new_signal = -POSITION_SIZE_BASE
                if crsi_overbought:
                    new_signal = -POSITION_SIZE_MAX
        
        # === HOLD POSITION LOGIC ===
        if in_position and new_signal == 0.0:
            if position_side > 0 and rsi_14[i] < 70.0:
                new_signal = signals[i-1] if i > 0 else 0.0
            elif position_side < 0 and rsi_14[i] > 30.0:
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
        
        # === EXIT ON TREND CHANGE ===
        if in_position and position_side > 0:
            if price_below_hma_1w and hma_slope_negative:
                new_signal = 0.0
        
        if in_position and position_side < 0:
            if price_above_hma_1w and hma_slope_positive:
                new_signal = 0.0
        
        # === EXIT ON RSI EXTREME (take profit) ===
        if in_position and position_side > 0 and rsi_14[i] > 75.0:
            new_signal = 0.0
        
        if in_position and position_side < 0 and rsi_14[i] < 25.0:
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
2026-03-23 04:51
