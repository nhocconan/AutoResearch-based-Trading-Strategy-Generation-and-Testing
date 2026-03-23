#!/usr/bin/env python3
"""
Experiment #465: 1h Primary + 4h/1d HTF — Regime-Adaptive CRSI/Fisher + Session Filter

Hypothesis: Based on research showing regime-adaptive strategies outperform static approaches
in crypto's alternating bull/bear/range markets. Key innovations:
1. CONNORS RSI (CRSI) for mean reversion in range markets (75% win rate documented)
   CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
2. ADX(14) regime detection with hysteresis: ADX>25=trend, ADX<18=range, 18-25=transition
3. 4h HMA(21) for primary trend bias (not 1d - too slow for 1h entries)
4. 1d ADX for confirming regime strength
5. Session filter: only 8-20 UTC (London/NY overlap = highest liquidity)
6. Volume filter: volume > 0.8x 20-bar rolling median
7. Asymmetric entries: CRSI<10 long in range, CRSI>90 short in range
8. Fisher Transform for trend entries when ADX>25
9. Strict 2.5x ATR trailing stoploss
10. Discrete sizing: 0.20 base, 0.30 max (lower for 1h to reduce fee drag)

Target: Sharpe > 0.612, 30-60 trades/year, DD < -35%
Timeframe: 1h (with strict filters to minimize fee drag)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_regime_crsi_fisher_session_4h1d_v1"
timeframe = "1h"
leverage = 1.0

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    Research shows 75% win rate for CRSI<10 long, CRSI>90 short entries.
    """
    n = len(close)
    crsi = np.full(n, np.nan)
    
    if n < rank_period:
        return crsi
    
    # RSI(3)
    delta = np.diff(close)
    delta = np.insert(delta, 0, 0)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    gain_rsi = pd.Series(gain).ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    loss_rsi = pd.Series(loss).ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = gain_rsi / (loss_rsi + 1e-10)
        rsi_3 = 100.0 - (100.0 / (1.0 + rs))
    rsi_3 = np.clip(rsi_3, 0, 100)
    
    # RSI Streak (consecutive up/down days)
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    # Convert streak to RSI-like value (0-100)
    streak_rsi = np.full(n, np.nan)
    for i in range(streak_period, n):
        streak_vals = streak[max(0, i-streak_period+1):i+1]
        up_streaks = np.sum(streak_vals > 0)
        down_streaks = np.sum(streak_vals < 0)
        total = up_streaks + down_streaks
        if total > 0:
            streak_rsi[i] = 100.0 * up_streaks / total
        else:
            streak_rsi[i] = 50.0
    
    # Percent Rank (100)
    percent_rank = np.full(n, np.nan)
    for i in range(rank_period, n):
        window = close[i-rank_period+1:i+1]
        current = close[i]
        count_below = np.sum(window[:-1] < current)
        percent_rank[i] = 100.0 * count_below / (rank_period - 1)
    
    # Combine into CRSI
    valid_mask = ~np.isnan(rsi_3) & ~np.isnan(streak_rsi) & ~np.isnan(percent_rank)
    crsi[valid_mask] = (rsi_3[valid_mask] + streak_rsi[valid_mask] + percent_rank[valid_mask]) / 3.0
    crsi = np.clip(crsi, 0, 100)
    
    return crsi

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index) using Wilder's smoothing."""
    n = len(close)
    adx = np.full(n, np.nan)
    
    if n < period * 2:
        return adx
    
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Directional Movement
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        plus_diff = high[i] - high[i-1]
        minus_diff = low[i-1] - low[i]
        
        if plus_diff > minus_diff and plus_diff > 0:
            plus_dm[i] = plus_diff
        if minus_diff > plus_diff and minus_diff > 0:
            minus_dm[i] = minus_diff
    
    # Smooth with Wilder's method (EMA with alpha = 1/period)
    tr_smooth = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # Directional Indicators
    with np.errstate(divide='ignore', invalid='ignore'):
        plus_di = 100.0 * plus_dm_smooth / (tr_smooth + 1e-10)
        minus_di = 100.0 * minus_dm_smooth / (tr_smooth + 1e-10)
    
    # DX
    dx = 100.0 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    
    # ADX (smoothed DX)
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx, plus_di, minus_di

def calculate_fisher_transform(high, low, period=9):
    """Calculate Ehlers Fisher Transform for reversal detection."""
    n = len(high)
    fisher = np.full(n, np.nan)
    trigger = np.full(n, np.nan)
    
    for i in range(period, n):
        highest = np.nanmax(high[i-period+1:i+1])
        lowest = np.nanmin(low[i-period+1:i+1])
        price_range = highest - lowest
        
        if price_range > 1e-10:
            x = (2.0 * (high[i] + low[i]) / 2.0 - highest - lowest) / price_range
            x = np.clip(x, -0.999, 0.999)
            
            fisher_val = 0.5 * np.log((1.0 + x) / (1.0 - x))
            
            if i > period and not np.isnan(fisher[i-1]):
                fisher[i] = 0.67 * fisher_val + 0.33 * fisher[i-1]
            else:
                fisher[i] = fisher_val
            
            if i > 0 and not np.isnan(fisher[i-1]):
                trigger[i] = fisher[i-1]
    
    return fisher, trigger

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

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def get_hour_from_open_time(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)."""
    return (open_time // 3600000) % 24

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1h indicators (primary timeframe)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    adx_1h, plus_di_1h, minus_di_1h = calculate_adx(high, low, close, period=14)
    fisher, fisher_trigger = calculate_fisher_transform(high, low, period=9)
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Volume rolling median for filter
    vol_median = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    
    # Calculate and align HTF indicators
    hma_4h_raw = calculate_hma(df_4h['close'].values, 21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    adx_1d_raw, _, _ = calculate_adx(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, period=14)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d_raw)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.20  # Conservative for 1h (lower fee drag)
    MAX_SIZE = 0.30
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    # ADX hysteresis tracking
    prev_adx_regime = 0  # 0=unknown, 1=trend, 2=range
    
    for i in range(300, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            continue
        if np.isnan(crsi[i]) or np.isnan(adx_1h[i]):
            continue
        if np.isnan(fisher[i]) or np.isnan(hma_4h_aligned[i]):
            continue
        if np.isnan(adx_1d_aligned[i]) or np.isnan(vol_median[i]):
            continue
        
        # === SESSION FILTER (8-20 UTC only) ===
        hour = get_hour_from_open_time(open_time[i])
        in_session = 8 <= hour <= 20
        
        # === VOLUME FILTER ===
        vol_ok = volume[i] > 0.8 * vol_median[i] if vol_median[i] > 0 else False
        
        # === REGIME DETECTION with HYSTERESIS ===
        adx_val = adx_1h[i]
        adx_1d_val = adx_1d_aligned[i]
        
        # Hysteresis: enter trend at 25, exit at 18
        if prev_adx_regime == 1:  # Was in trend
            is_trend = adx_val > 18
        elif prev_adx_regime == 2:  # Was in range
            is_trend = adx_val > 25
        else:  # Unknown
            is_trend = adx_val > 25
        
        is_range = not is_trend
        prev_adx_regime = 1 if is_trend else 2
        
        # Confirm with 1d ADX (avoid false signals)
        if not np.isnan(adx_1d_val):
            if is_trend and adx_1d_val < 20:
                is_trend = False  # 1d says weak trend
                is_range = True
            if is_range and adx_1d_val > 30:
                is_range = False  # 1d says strong trend
                is_trend = True
        
        # === HTF TREND BIAS (4h HMA) ===
        price_above_hma_4h = close[i] > hma_4h_aligned[i]
        price_below_hma_4h = close[i] < hma_4h_aligned[i]
        
        # === CONNORS RSI SIGNALS ===
        crsi_oversold = crsi[i] < 12  # Slightly wider than 10 for more trades
        crsi_overbought = crsi[i] > 88  # Slightly narrower than 90
        
        # === FISHER TRANSFORM SIGNALS ===
        fisher_oversold = fisher[i] < -1.2
        fisher_overbought = fisher[i] > 1.2
        fisher_cross_up = fisher[i] > fisher_trigger[i] if not np.isnan(fisher_trigger[i]) else False
        fisher_cross_down = fisher[i] < fisher_trigger[i] if not np.isnan(fisher_trigger[i]) else False
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        # Only trade in session with volume
        if not (in_session and vol_ok):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === REGIME 1: RANGE (ADX < threshold) — MEAN REVERSION with CRSI ===
        if is_range:
            # Long: CRSI oversold + price near/above 4h HMA (not in strong downtrend)
            if crsi_oversold:
                confluence = 1
                if price_above_hma_4h:
                    confluence += 1
                if fisher_oversold or fisher_cross_up:
                    confluence += 1
                
                if confluence >= 2:
                    size = BASE_SIZE + 0.05 * min(confluence - 2, 1)
                    desired_signal = min(size, MAX_SIZE)
            
            # Short: CRSI overbought + price near/below 4h HMA
            if crsi_overbought and desired_signal == 0:
                confluence = 1
                if price_below_hma_4h:
                    confluence += 1
                if fisher_overbought or fisher_cross_down:
                    confluence += 1
                
                if confluence >= 2:
                    size = BASE_SIZE + 0.05 * min(confluence - 2, 1)
                    desired_signal = -min(size, MAX_SIZE)
        
        # === REGIME 2: TREND (ADX > threshold) — TREND FOLLOW with Fisher ===
        elif is_trend:
            # Long: Fisher cross up + price above 4h HMA + DI+ > DI-
            if fisher_cross_up and price_above_hma_4h:
                confluence = 2
                if plus_di_1h[i] > minus_di_1h[i]:
                    confluence += 1
                if not np.isnan(adx_1d_val) and adx_1d_val > 25:
                    confluence += 1
                
                if confluence >= 2:
                    desired_signal = BASE_SIZE
            
            # Short: Fisher cross down + price below 4h HMA + DI- > DI+
            if fisher_cross_down and price_below_hma_4h and desired_signal == 0:
                confluence = 2
                if minus_di_1h[i] > plus_di_1h[i]:
                    confluence += 1
                if not np.isnan(adx_1d_val) and adx_1d_val > 25:
                    confluence += 1
                
                if confluence >= 2:
                    desired_signal = -BASE_SIZE
        
        # === CAP SIGNAL ===
        if desired_signal > MAX_SIZE:
            desired_signal = MAX_SIZE
        elif desired_signal < -MAX_SIZE:
            desired_signal = -MAX_SIZE
        
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
        
        # === FISHER EXTREME EXIT (Take Profit) ===
        if in_position and position_side > 0 and fisher[i] > 1.5:
            desired_signal = 0.0
        
        if in_position and position_side < 0 and fisher[i] < -1.5:
            desired_signal = 0.0
        
        # === CRSI EXTREME EXIT ===
        if in_position and position_side > 0 and crsi[i] > 80:
            desired_signal = 0.0
        
        if in_position and position_side < 0 and crsi[i] < 20:
            desired_signal = 0.0
        
        # === HOLD LOGIC — Maintain position if bias unchanged ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0 and (price_above_hma_4h or crsi[i] < 70):
                desired_signal = BASE_SIZE
            elif position_side < 0 and (price_below_hma_4h or crsi[i] > 30):
                desired_signal = -BASE_SIZE
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal != 0.0:
            if desired_signal > 0:
                if desired_signal >= 0.28:
                    desired_signal = 0.30
                elif desired_signal >= 0.18:
                    desired_signal = 0.20
                else:
                    desired_signal = 0.15
            else:
                if desired_signal <= -0.28:
                    desired_signal = -0.30
                elif desired_signal <= -0.18:
                    desired_signal = -0.20
                else:
                    desired_signal = -0.15
        
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