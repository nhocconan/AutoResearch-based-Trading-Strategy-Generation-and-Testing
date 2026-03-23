#!/usr/bin/env python3
"""
Experiment #1418: 30m Primary + 4h/1d HTF — Choppiness Regime + Connors RSI + Volume Session

Hypothesis: 30m timeframe with 4h trend filter can work IF we use strict confluence to limit
trade frequency to 40-80/year. The key insight from failed 30m/1h strategies (#1408, #1410, #1415)
is they generated 0 trades due to overly strict conditions. We need to balance:
- Strict enough to limit trades (avoid fee drag)
- Loose enough to actually trigger entries (avoid 0 trades = auto reject)

Design:
1. 4h HMA(21) = trend direction (call ONCE before loop via mtf_data)
2. 1d HMA(21) = macro filter for additional confluence
3. Choppiness Index(14) = regime detection (>55 range, <45 trend)
4. Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
5. Volume filter = volume > 0.7x 20-period average (soft filter)
6. Session filter = 8-20 UTC only (avoid low liquidity Asian night)
7. Long: 4h_HMA bull + (CHOP>55+CRSI<25 OR CHOP<45+breakout) + volume + session
8. Short: 4h_HMA bear + (CHOP>55+CRSI>75 OR CHOP<45+breakdown) + volume + session
9. ATR(14) trailing stop 2.5x = risk management
10. Position size 0.25 = conservative for 30m volatility

Target: 50-100 trades/year, Sharpe > 0.618, trades >= 30 train, >= 5 test
Timeframe: 30m
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_chop_regime_crsi_4h1d_hma_vol_session_atr_v1"
timeframe = "30m"
leverage = 1.0

def calculate_hma(close, period=21):
    """Hull Moving Average - faster response than EMA, less lag"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    def wma(series, span):
        weights = np.arange(1, span + 1, dtype=np.float64)
        result = np.full(len(series), np.nan)
        for i in range(span - 1, len(series)):
            window = series[i - span + 1:i + 1]
            if not np.any(np.isnan(window)):
                result[i] = np.sum(window * weights) / np.sum(weights)
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    hma = np.full(n, np.nan)
    for i in range(period - 1, n):
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            diff = 2.0 * wma_half[i] - wma_full[i]
            if i >= sqrt_period - 1:
                diff_vals = []
                for j in range(i - sqrt_period + 1, i + 1):
                    if j >= period - 1 and not np.isnan(2.0 * wma_half[j] - wma_full[j]):
                        diff_vals.append(2.0 * wma_half[j] - wma_full[j])
                if len(diff_vals) == sqrt_period:
                    weights = np.arange(1, sqrt_period + 1, dtype=np.float64)
                    hma[i] = np.sum(np.array(diff_vals) * weights) / np.sum(weights)
    
    return hma

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    gain_smooth = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    loss_smooth = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.full(n, np.nan)
    mask = loss_smooth > 1e-10
    rsi[mask] = 100.0 - (100.0 / (1.0 + gain_smooth[mask] / loss_smooth[mask]))
    rsi[loss_smooth <= 1e-10] = 100.0
    rsi[:period] = np.nan
    
    return rsi

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3"""
    n = len(close)
    if n < rank_period:
        return np.full(n, np.nan)
    
    crsi = np.full(n, np.nan)
    
    # RSI(3)
    rsi_short = calculate_rsi(close, rsi_period)
    
    # RSI Streak (consecutive up/down days)
    streak_rsi = np.full(n, np.nan)
    for i in range(streak_period, n):
        streak = 0
        if i > 0:
            if close[i] > close[i-1]:
                j = i
                while j > 0 and close[j] > close[j-1]:
                    streak += 1
                    j -= 1
            elif close[i] < close[i-1]:
                j = i
                while j > 0 and close[j] < close[j-1]:
                    streak -= 1
                    j -= 1
        # Convert streak to RSI-like value (0-100)
        if streak >= 0:
            streak_rsi[i] = min(100.0, streak * 50.0 / streak_period)
        else:
            streak_rsi[i] = max(0.0, 100.0 + streak * 50.0 / streak_period)
    
    # Percent Rank (100)
    percent_rank = np.full(n, np.nan)
    for i in range(rank_period, n):
        returns = np.diff(close[i-rank_period+1:i+1])
        if len(returns) > 0 and not np.any(np.isnan(returns)):
            current_return = returns[-1]
            count_below = np.sum(returns[:-1] < current_return)
            percent_rank[i] = 100.0 * count_below / (len(returns) - 1) if len(returns) > 1 else 50.0
    
    # Combine
    for i in range(rank_period, n):
        if not np.isnan(rsi_short[i]) and not np.isnan(streak_rsi[i]) and not np.isnan(percent_rank[i]):
            crsi[i] = (rsi_short[i] + streak_rsi[i] + percent_rank[i]) / 3.0
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """Choppiness Index - measures market choppy vs trending"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    chop = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        highest_high = np.nanmax(high[i-period+1:i+1])
        lowest_low = np.nanmin(low[i-period+1:i+1])
        
        if highest_high > lowest_low:
            tr_sum = 0.0
            for j in range(i-period+1, i+1):
                tr_sum += max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
            
            chop[i] = 100.0 * np.log10(tr_sum / (highest_high - lowest_low)) / np.log10(period)
    
    return chop

def calculate_atr(high, low, close, period=14):
    """Average True Range - for stoploss sizing"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_volume_ma(volume, period=20):
    """Simple moving average of volume"""
    n = len(volume)
    if n < period:
        return np.full(n, np.nan)
    
    vol_ma = np.full(n, np.nan)
    for i in range(period - 1, n):
        vol_ma[i] = np.nanmean(volume[i-period+1:i+1])
    
    return vol_ma

def calculate_donchian(high, low, period=20):
    """Donchian Channel - breakout levels for entry trigger"""
    n = len(high)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        upper[i] = np.nanmax(high[i-period+1:i+1])
        lower[i] = np.nanmin(low[i-period+1:i+1])
    
    return upper, lower

def get_utc_hour(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)"""
    # open_time is in milliseconds since epoch
    import datetime
    dt = datetime.datetime.utcfromtimestamp(open_time / 1000.0)
    return dt.hour

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
    
    # Calculate and align 4h HMA for trend direction
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate and align 1d HMA for macro filter
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (30m) indicators
    donchian_20_upper, donchian_20_lower = calculate_donchian(high, low, period=20)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    chop = calculate_choppiness(high, low, close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    vol_ma = calculate_volume_ma(volume, period=20)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.25
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(donchian_20_upper[i]) or np.isnan(chop[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(crsi[i]) or np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === SESSION FILTER (8-20 UTC) ===
        utc_hour = get_utc_hour(open_time[i])
        in_session = 8 <= utc_hour <= 20
        
        # === VOLUME FILTER (soft - 0.7x average) ===
        vol_ok = True
        if not np.isnan(vol_ma[i]) and vol_ma[i] > 1e-10:
            vol_ok = volume[i] > 0.7 * vol_ma[i]
        
        # === TREND FILTERS (4h + 1d HMA) ===
        trend_4h_bull = close[i] > hma_4h_aligned[i]
        trend_4h_bear = close[i] < hma_4h_aligned[i]
        
        # 1d HMA as additional confluence (not required but adds confidence)
        trend_1d_bull = not np.isnan(hma_1d_aligned[i]) and close[i] > hma_1d_aligned[i]
        trend_1d_bear = not np.isnan(hma_1d_aligned[i]) and close[i] < hma_1d_aligned[i]
        
        # === CHOPPINESS REGIME ===
        is_choppy = chop[i] > 55.0  # Range market
        is_trending = chop[i] < 45.0  # Trend market
        
        # === CONNORS RSI EXTREMES (mean reversion in chop) ===
        crsi_oversold = crsi[i] < 30.0  # Relaxed from 25 for more trades
        crsi_overbought = crsi[i] > 70.0  # Relaxed from 75 for more trades
        
        # === DONCHIAN BREAKOUT (trend following) ===
        breakout_20_long = close[i] > donchian_20_upper[i-1] if i > 0 else False
        breakout_20_short = close[i] < donchian_20_lower[i-1] if i > 0 else False
        
        # === DESIRED SIGNAL - DUAL REGIME LOGIC ===
        desired_signal = 0.0
        
        # LONG ENTRIES - need 4h bull trend + confluence
        if trend_4h_bull:
            # Path 1: Choppy regime + CRSI oversold + volume (mean reversion)
            if is_choppy and crsi_oversold and vol_ok:
                desired_signal = BASE_SIZE
            # Path 2: Trending regime + Donchian-20 breakout + volume
            elif is_trending and breakout_20_long and vol_ok:
                desired_signal = BASE_SIZE
            # Path 3: CRSI very oversold (stronger signal, volume optional)
            elif crsi[i] < 20.0:
                desired_signal = BASE_SIZE * 0.5
        
        # SHORT ENTRIES - need 4h bear trend + confluence
        elif trend_4h_bear:
            # Path 1: Choppy regime + CRSI overbought + volume (mean reversion)
            if is_choppy and crsi_overbought and vol_ok:
                desired_signal = -BASE_SIZE
            # Path 2: Trending regime + Donchian-20 breakdown + volume
            elif is_trending and breakout_20_short and vol_ok:
                desired_signal = -BASE_SIZE
            # Path 3: CRSI very overbought (stronger signal, volume optional)
            elif crsi[i] > 80.0:
                desired_signal = -BASE_SIZE * 0.5
        
        # Apply session filter (only enter during liquid hours)
        if not in_session and desired_signal != 0.0:
            # Don't enter new positions outside session, but don't exit existing
            if not in_position:
                desired_signal = 0.0
        
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
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= BASE_SIZE * 0.4:
            final_signal = BASE_SIZE if desired_signal > 0 else -BASE_SIZE
        elif desired_signal <= -BASE_SIZE * 0.4:
            final_signal = -BASE_SIZE
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(final_signal) != position_side:
                # Flip position
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
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
        
        signals[i] = final_signal
    
    return signals