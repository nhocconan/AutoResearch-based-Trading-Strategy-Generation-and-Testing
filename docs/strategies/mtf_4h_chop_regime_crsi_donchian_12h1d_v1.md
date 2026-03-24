# Strategy: mtf_4h_chop_regime_crsi_donchian_12h1d_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.027 | +17.3% | -12.8% | 490 | FAIL |
| ETHUSDT | 0.214 | +32.7% | -15.0% | 492 | PASS |
| SOLUSDT | 0.540 | +81.6% | -32.5% | 532 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.801 | +22.3% | -12.3% | 160 | PASS |
| SOLUSDT | -0.214 | -0.4% | -19.8% | 170 | FAIL |

## Code
```python
#!/usr/bin/env python3
"""
Experiment #434: 4h Primary + 12h/1d HTF — Choppiness Regime Switch + Dual Strategy

Hypothesis: Single-strategy approaches fail because crypto alternates between trending and ranging.
By using Choppiness Index as a REGIME SWITCH, we can:
1. Use CRSI mean reversion when CHOP > 61.8 (range market)
2. Use Donchian/HMA trend follow when CHOP < 38.2 (trending market)
3. Stay flat or reduce size in transition zones (38.2-61.8)

This dual-regime approach should:
- Generate MORE trades than #431 (both mean reversion AND trend entries)
- Work in BOTH bull and bear markets (adaptive to regime)
- Beat Sharpe=0.612 by reducing whipsaw losses in wrong regime

Key improvements over #431:
- Choppiness Index regime detection (proven on ETH Sharpe +0.923)
- Simpler entry conditions per regime (avoid 0-trade failure)
- 12h HMA bias (faster response than 1d, less lag)
- Position size 0.28 (conservative for 4h)

Target: Sharpe > 0.612, 100-200 trades over 4-year train, DD < -40%
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_chop_regime_crsi_donchian_12h1d_v1"
timeframe = "4h"
leverage = 1.0

def calculate_hma(close, period):
    """Calculate Hull Moving Average."""
    close_s = pd.Series(close)
    half = period // 2
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    diff = 2.0 * wma1 - wma2
    sqrt_period = int(np.sqrt(period))
    hma = diff.ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return hma.values

def calculate_rsi(close, period):
    """Calculate RSI using Wilder's smoothing."""
    n = len(close)
    rsi = np.full(n, np.nan)
    
    delta = np.diff(close)
    delta = np.insert(delta, 0, 0)
    
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    gain_s = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    loss_s = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = gain_s / (loss_s + 1e-10)
        rsi = 100.0 - (100.0 / (1.0 + rs))
    
    rsi = np.clip(rsi, 0, 100)
    return rsi

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    
    CHOP > 61.8 = choppy/ranging market (mean reversion works)
    CHOP < 38.2 = trending market (trend following works)
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    atr = calculate_atr(high, low, close, period)
    
    for i in range(period, n):
        sum_atr = np.nansum(atr[i-period+1:i+1])
        highest_high = np.nanmax(high[i-period+1:i+1])
        lowest_low = np.nanmin(low[i-period+1:i+1])
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10 and sum_atr > 0:
            chop[i] = 100.0 * np.log10(sum_atr / price_range) / np.log10(period)
        else:
            chop[i] = 50.0  # neutral
    
    chop = np.clip(chop, 0, 100)
    return chop

def calculate_crsi_component_rsi(close, period=3):
    """RSI(3) component of CRSI."""
    return calculate_rsi(close, period)

def calculate_crsi_component_streak(close, period=2):
    """
    Calculate RSI Streak component of CRSI.
    Measures consecutive up/down bars.
    """
    n = len(close)
    streak_rsi = np.full(n, np.nan)
    
    for i in range(period, n):
        up_streak = 0
        down_streak = 0
        
        for j in range(i, max(i - period - 5, 0), -1):
            if j == 0:
                break
            if close[j] > close[j-1]:
                up_streak += 1
                down_streak = 0
            elif close[j] < close[j-1]:
                down_streak += 1
                up_streak = 0
            else:
                up_streak = 0
                down_streak = 0
                break
        
        streak = up_streak if up_streak > 0 else -down_streak
        
        # Convert streak to RSI-like scale
        if streak > 0:
            streak_rsi[i] = 50.0 + (streak / (period + 1)) * 50.0
        elif streak < 0:
            streak_rsi[i] = 50.0 - (abs(streak) / (period + 1)) * 50.0
        else:
            streak_rsi[i] = 50.0
    
    streak_rsi = np.clip(streak_rsi, 0, 100)
    return streak_rsi

def calculate_crsi_component_percent_rank(close, period=100):
    """
    Calculate Percent Rank component of CRSI.
    Percentage of past returns less than current return.
    """
    n = len(close)
    pct_rank = np.full(n, np.nan)
    
    returns = np.diff(close) / (close[:-1] + 1e-10)
    returns = np.insert(returns, 0, 0)
    
    for i in range(period, n):
        window = returns[i-period+1:i+1]
        current = returns[i]
        pct_rank[i] = 100.0 * np.sum(window < current) / period
    
    return pct_rank

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    """
    rsi = calculate_crsi_component_rsi(close, rsi_period)
    streak_rsi = calculate_crsi_component_streak(close, streak_period)
    pct_rank = calculate_crsi_component_percent_rank(close, rank_period)
    
    with np.errstate(invalid='ignore'):
        crsi = (rsi + streak_rsi + pct_rank) / 3.0
    
    return crsi

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (highest high / lowest low over period)."""
    n = len(high)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        upper[i] = np.nanmax(high[i-period+1:i+1])
        lower[i] = np.nanmin(low[i-period+1:i+1])
    
    return upper, lower

def calculate_sma(close, period):
    """Calculate Simple Moving Average."""
    close_s = pd.Series(close)
    return close_s.rolling(window=period, min_periods=period).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 4h indicators (primary timeframe)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    hma_21 = calculate_hma(close, 21)
    hma_50 = calculate_hma(close, 50)
    chop = calculate_choppiness(high, low, close, period=14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    atr_14 = calculate_atr(high, low, close, period=14)
    sma_200 = calculate_sma(close, 200)
    
    # Calculate and align HTF HMA for bias (12h and 1d)
    hma_12h_raw = calculate_hma(df_12h['close'].values, 21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    hma_1d_raw = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate median ATR for vol filter
    valid_atr = atr_14[200:]
    atr_median = np.nanmedian(valid_atr[~np.isnan(valid_atr)])
    if np.isnan(atr_median) or atr_median <= 0:
        atr_median = np.nanmean(valid_atr[~np.isnan(valid_atr)])
    
    signals = np.zeros(n)
    BASE_SIZE = 0.28  # 28% position size for 4h
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(250, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            continue
        if np.isnan(crsi[i]) or np.isnan(chop[i]):
            continue
        if np.isnan(hma_21[i]) or np.isnan(sma_200[i]):
            continue
        if np.isnan(hma_12h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            continue
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            continue
        
        # === REGIME DETECTION (Choppiness Index) ===
        regime_chop = chop[i] > 61.8  # Range market → mean reversion
        regime_trend = chop[i] < 38.2  # Trending market → trend follow
        regime_transition = not regime_chop and not regime_trend  # 38.2-61.8
        
        # === TREND BIAS (12h HMA + 1d HMA) ===
        price_above_hma_12h = close[i] > hma_12h_aligned[i]
        price_below_hma_12h = close[i] < hma_12h_aligned[i]
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        price_below_hma_1d = close[i] < hma_1d_aligned[i]
        
        # === PRIMARY TREND (4h HMA) ===
        hma_bullish = hma_21[i] > hma_50[i]
        hma_bearish = hma_21[i] < hma_50[i]
        
        # === CRSI SIGNALS (Mean Reversion) ===
        crsi_oversold = crsi[i] < 20.0  # Long entry threshold (relaxed from 15)
        crsi_overbought = crsi[i] > 80.0  # Short entry threshold (relaxed from 85)
        crsi_extreme_oversold = crsi[i] < 10.0
        crsi_extreme_overbought = crsi[i] > 90.0
        
        # === DONCHIAN BREAKOUT (Trend Follow) ===
        donchian_breakout_long = close[i] > donchian_upper[i-1] if not np.isnan(donchian_upper[i-1]) else False
        donchian_breakout_short = close[i] < donchian_lower[i-1] if not np.isnan(donchian_lower[i-1]) else False
        
        # === VOL FILTER ===
        vol_ratio = atr_14[i] / (atr_median + 1e-10)
        if vol_ratio > 2.5:
            position_size = BASE_SIZE * 0.5
        elif vol_ratio > 1.8:
            position_size = BASE_SIZE * 0.7
        else:
            position_size = BASE_SIZE
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        # === REGIME 1: CHOPPY/RANGE (CHOP > 61.8) — MEAN REVERSION ===
        if regime_chop:
            # Long: CRSI oversold + price above longer-term trend
            if crsi_oversold and (price_above_hma_12h or price_above_hma_1d or close[i] > sma_200[i]):
                desired_signal = position_size
            elif crsi_extreme_oversold:  # Extreme oversold = strong long regardless
                desired_signal = position_size * 1.2
            
            # Short: CRSI overbought + price below longer-term trend
            if crsi_overbought and (price_below_hma_12h or price_below_hma_1d or close[i] < sma_200[i]):
                desired_signal = -position_size
            elif crsi_extreme_overbought:  # Extreme overbought = strong short regardless
                desired_signal = -position_size * 1.2
        
        # === REGIME 2: TRENDING (CHOP < 38.2) — TREND FOLLOW ===
        elif regime_trend:
            # Long: Donchian breakout + trend bias
            if donchian_breakout_long and (hma_bullish or price_above_hma_12h):
                desired_signal = position_size
            elif hma_bullish and price_above_hma_12h and close[i] > sma_200[i]:
                desired_signal = position_size * 0.8
            
            # Short: Donchian breakdown + trend bias
            if donchian_breakout_short and (hma_bearish or price_below_hma_12h):
                desired_signal = -position_size
            elif hma_bearish and price_below_hma_12h and close[i] < sma_200[i]:
                desired_signal = -position_size * 0.8
        
        # === REGIME 3: TRANSITION (38.2-61.8) — REDUCED SIZE OR FLAT ===
        elif regime_transition:
            # Only take extreme CRSI signals with reduced size
            if crsi_extreme_oversold and (price_above_hma_12h or close[i] > sma_200[i]):
                desired_signal = position_size * 0.5
            elif crsi_extreme_overbought and (price_below_hma_12h or close[i] < sma_200[i]):
                desired_signal = -position_size * 0.5
        
        # === STOPLOSS CHECK (Trailing ATR) ===
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
        
        # === CRSI EXTREME EXIT (Take Profit) ===
        if in_position and position_side > 0 and crsi[i] > 85.0:
            desired_signal = 0.0
        
        if in_position and position_side < 0 and crsi[i] < 15.0:
            desired_signal = 0.0
        
        # === HTF BIAS REVERSAL EXIT ===
        if in_position and position_side > 0 and price_below_hma_1d and hma_bearish:
            desired_signal = 0.0
        
        if in_position and position_side < 0 and price_above_hma_1d and hma_bullish:
            desired_signal = 0.0
        
        # === HOLD LOGIC — Maintain position if bias unchanged ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0 and (price_above_hma_12h or hma_bullish or close[i] > sma_200[i]):
                desired_signal = position_size
            elif position_side < 0 and (price_below_hma_12h or hma_bearish or close[i] < sma_200[i]):
                desired_signal = -position_size
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
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
2026-03-23 10:25
