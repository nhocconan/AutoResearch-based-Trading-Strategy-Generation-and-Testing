#!/usr/bin/env python3
"""
Experiment #028: 1d CRSI + Weekly EMA21 Trend + Choppiness Regime

HYPOTHESIS: CRSI (Connors RSI) identifies short-term mean reversion extremes 
within sustained trends. By combining CRSI extremes (<15 long, >85 short) with 
weekly EMA21 for trend direction and Choppiness to filter range-bound markets,
we capture high-probability reversals aligned with the broader trend.

WHY 1d: Slower than 4h/6h = fewer false signals = lower fee drag.
Weekly EMA21 = 5-week moving average for reliable trend direction.
CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3 — proven edge.

WHY IT WORKS IN BULL AND BEAR:
- Bull: CRSI < 15 catches oversold dips that bounce. Trend stays up = high win rate.
- Bear: CRSI > 85 catches overbought rallies that fade. Trend stays down = short winners.
- Choppiness keeps us out during transitions (2022 bottom whipsaw protection).
- Weekly EMA21 changes ~6-8 times/year = reliable trend filter without overtrading.

TARGET: 50-100 total trades over 4 years = 12-25/year. HARD MAX: 150.
Signal size: 0.30.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_crsi_1w_ema21_chop_v1"
timeframe = "1d"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """Connors RSI: (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3"""
    n = len(close)
    
    # RSI(3) using EWM
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean()
    avg_loss = loss.ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean()
    rs = avg_gain / np.where(avg_loss == 0, 1e-10, avg_loss)
    rsi = 100 - (100 / (1 + rs))
    
    # RSI Streak(2) - consecutive up/down closes
    streak = np.zeros(n)
    current_streak = 0
    for i in range(1, n):
        if close[i] > close[i-1]:
            current_streak = max(0, current_streak + 1)
        elif close[i] < close[i-1]:
            current_streak = min(0, current_streak - 1)
        streak[i] = current_streak
    
    # RSI of streaks
    streak_series = pd.Series(streak)
    streak_gain = streak_series.clip(lower=0)
    streak_loss = -streak_series.clip(upper=0)
    streak_avg_gain = streak_gain.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    streak_avg_loss = streak_loss.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    streak_rs = streak_avg_gain / np.where(streak_avg_loss == 0, 1e-10, streak_avg_loss)
    rsi_streak = 100 - (100 / (1 + streak_rs))
    
    # PercentRank(100) using rolling percentile
    def rolling_percent_rank(x):
        return (x[-1] < x).sum() / len(x)
    
    percent_rank = pd.Series(close).rolling(window=rank_period, min_periods=rank_period).apply(
        rolling_percent_rank, raw=True
    )
    
    # CRSI = average of all three
    crsi = (rsi + rsi_streak + percent_rank) / 3
    
    return crsi.values

def calculate_choppiness(high, low, close, period=14):
    """Choppiness Index - lower = trending, higher = choppy"""
    n = len(high)
    chop = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period, n):
        tr_sum = 0.0
        for j in range(i - period + 1, i + 1):
            if j > 0:
                tr = max(high[j] - low[j], abs(high[j] - close[j-1]))
            else:
                tr = high[j] - low[j]
            tr_sum += tr
        
        if tr_sum > 0:
            hh = np.max(high[i - period + 1:i + 1])
            ll = np.min(low[i - period + 1:i + 1])
            range_hl = hh - ll
            
            if range_hl > 0:
                chop[i] = 100 * np.log10(tr_sum / range_hl) / np.log10(period)
    
    return chop

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE ===
    df_1w = get_htf_data(prices, '1w')
    
    # 1w EMA21 for trend direction
    ema_1w = pd.Series(df_1w['close'].values).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Local 1d indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    chop = calculate_choppiness(high, low, close, period=14)
    
    # Volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    signals = np.zeros(n)
    SIZE = 0.30
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_bar = 0
    
    warmup = 120  # Need enough for CRSI(100) + buffer
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(crsi[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(chop[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(ema_1w_aligned[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # === TREND DIRECTION (1w EMA21) ===
        weekly_trend_bullish = close[i] > ema_1w_aligned[i]
        weekly_trend_bearish = close[i] < ema_1w_aligned[i]
        
        # === REGIME (Choppiness Index) ===
        # Skip if too choppy (avoid whipsaws)
        is_choppy = chop[i] > 61.8
        
        # Volume confirmation (optional boost)
        vol_spike = vol_ratio[i] > 1.5
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # === LONG: CRSI oversold + weekly bullish trend ===
            # CRSI < 15 = extreme oversold (rare ~5-10% of days)
            # Weekly trend up = higher probability bounce
            if crsi[i] < 15 and weekly_trend_bullish:
                # Optional volume confirmation (but not required)
                desired_signal = SIZE
            
            # === SHORT: CRSI overbought + weekly bearish trend ===
            # CRSI > 85 = extreme overbought (rare ~5-10% of days)
            # Weekly trend down = higher probability fade
            if crsi[i] > 85 and weekly_trend_bearish:
                # Optional volume confirmation (but not required)
                desired_signal = -SIZE
        
        # === STOPLOSS CHECK (2.5 ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.5 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.5 * entry_atr
            stop_price = min(stop_price, trailing_stop)
            if high[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === EXIT: CRSI mean reversion (CRSI crosses 50) ===
        if in_position and crsi[i] > 50 and position_side > 0:
            desired_signal = 0.0
        
        if in_position and crsi[i] < 50 and position_side < 0:
            desired_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                # New position or flip
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                entry_bar = i
                if position_side > 0:
                    stop_price = entry_price - 2.5 * entry_atr
                else:
                    stop_price = entry_price + 2.5 * entry_atr
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                stop_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = desired_signal
    
    return signals