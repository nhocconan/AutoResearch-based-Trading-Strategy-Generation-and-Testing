#!/usr/bin/env python3
"""
Experiment #020: 1d CRSI + Choppiness Regime + Donchian Breakout

HYPOTHESIS: This is a DIRECT COPY of the proven mtf_4h_crsi_chop_donchian_regime_1d_v1
which achieved test Sharpe 1.460 on SOLUSDT (392 trades, 39% win rate).
CRSI (Connors RSI) = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
- Long when CRSI < 10 (oversold) + price > SMA200 (bullish bias)
- Short when CRSI > 90 (overbought) + price < SMA200 (bearish bias)
- Choppiness Index > 61.8 = range market (crsi mean reversion works)
- Choppiness Index < 38.2 = trending (use Donchian for trend entries)
This captures reversals in ranges and breakouts in trends.

TIMEFRAME: 1d primary
HTF: 1w for regime/trend
TARGET: 75-200 total trades over 4 years (19-50/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_crsi_chop_donchian_regime_1w_v1"
timeframe = "1d"
leverage = 1.0

def calculate_rsi(close, period=14):
    """Standard RSI"""
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    return (100 - (100 / (1 + rs))).values

def calculate_rsi_streak(close, period=2):
    """RSI Streak - consecutive up/down closes"""
    n = len(close)
    deltas = pd.Series(close).diff().values
    
    streaks = np.zeros(n)
    current_streak = 0
    
    for i in range(1, n):
        if deltas[i] > 0:
            current_streak = max(0, current_streak) + 1
        elif deltas[i] < 0:
            current_streak = min(0, current_streak) - 1
        else:
            current_streak = 0
        streaks[i] = current_streak
    
    # Convert streaks to RSI-like values
    rsi_streak = np.full(n, 50.0)
    for i in range(period, n):
        window = streaks[max(0, i-period+1):i+1]
        if len(window) > 0:
            # Higher streak = more oversold (lower RSI streak value)
            avg_streak = np.mean(window)
            # Map to 0-100
            rsi_streak[i] = 50 - (avg_streak * 5)  # rough mapping
            rsi_streak[i] = np.clip(rsi_streak[i], 0, 100)
    
    return rsi_streak

def calculate_percent_rank(series, period=100):
    """PercentRank over rolling window"""
    n = len(series)
    pct_rank = np.full(n, 50.0)
    
    for i in range(period, n):
        window = series[max(0, i-period):i+1]
        if len(window) > 0 and not np.any(np.isnan(window)):
            current = series[i]
            rank = np.sum(window < current) / len(window)
            pct_rank[i] = rank * 100
    
    return pct_rank

def calculate_crsi(close, rsi_period=14, streak_period=2, rank_period=100):
    """Connors RSI = (RSI + RSI_Streak + PercentRank) / 3"""
    n = len(close)
    rsi3 = calculate_rsi(close, period=3)
    rsi_streak = calculate_rsi_streak(close, period=streak_period)
    pct_rank = calculate_percent_rank(close, period=rank_period)
    
    crsi = (rsi3 + rsi_streak + pct_rank) / 3.0
    return crsi

def calculate_choppiness(close, high, low, period=14):
    """
    Choppiness Index - values > 61.8 = choppy/range, < 38.2 = trending
    """
    n = len(close)
    chop = np.full(n, 61.8)  # default to range
    
    for i in range(period, n):
        if np.isnan(close[i]) or np.isnan(close[i-period]):
            continue
        
        # Sum of true range
        sum_tr = 0.0
        for j in range(i - period + 1, i + 1):
            tr = max(high[j] - low[j], 
                     abs(high[j] - close[j-1]) if j > 0 else high[j] - low[j],
                     abs(low[j] - close[j-1]) if j > 0 else high[j] - low[j])
            sum_tr += tr
        
        # Highest high - lowest low
        highest_high = np.max(high[i - period + 1:i + 1])
        lowest_low = np.min(low[i - period + 1:i + 1])
        
        range_sum = highest_high - lowest_low
        
        if range_sum > 0 and sum_tr > 0:
            chop[i] = 100 * np.log10(sum_tr / range_sum) / np.log10(period)
    
    return chop

def calculate_donchian(high, low, period=20):
    """Donchian Channel"""
    n = len(high)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    middle = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
        middle[i] = (upper[i] + lower[i]) / 2.0
    
    return upper, middle, lower

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], 
                    abs(high[i] - close[i-1]), 
                    abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # === Load HTF data ONCE ===
    df_1w = get_htf_data(prices, '1w')
    
    # 1w close for trend
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate 1w SMA200 equivalent for trend
    sma_1w_200 = pd.Series(close_1w).rolling(window=min(200, len(close_1w)), min_periods=200).mean().values if len(close_1w) >= 200 else pd.Series(close_1w).rolling(window=min(50, len(close_1w)), min_periods=50).mean().values
    sma_1w_aligned = align_htf_to_ltf(prices, df_1w, sma_1w_200)
    
    # 1w HMA for smoother trend
    def calc_hma(series, period):
        half = max(1, period // 2)
        sqrt_n = max(1, int(np.sqrt(period)))
        wma_half = pd.Series(series).rolling(window=half, min_periods=half).apply(
            lambda x: np.sum(x * np.arange(1, half+1)) / np.sum(np.arange(1, half+1)), raw=True).values
        wma_full = pd.Series(series).rolling(window=period, min_periods=period).apply(
            lambda x: np.sum(x * np.arange(1, period+1)) / np.sum(np.arange(1, period+1)), raw=True).values
        diff = 2 * wma_half - wma_full
        hma = pd.Series(diff).rolling(window=sqrt_n, min_periods=sqrt_n).apply(
            lambda x: np.sum(x * np.arange(1, sqrt_n+1)) / np.sum(np.arange(1, sqrt_n+1)), raw=True).values
        return hma
    
    hma_1w = calc_hma(close_1w, 21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 1d indicators
    crsi = calculate_crsi(close, rsi_period=14, streak_period=2, rank_period=100)
    chop = calculate_choppiness(close, high, low, period=14)
    upper_d, middle_d, lower_d = calculate_donchian(high, low, period=20)
    atr = calculate_atr(high, low, close, period=14)
    
    # Local SMA200
    sma_200 = pd.Series(close).rolling(window=200, min_periods=200).mean().values
    
    # Volume
    volume = prices["volume"].values
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    signals = np.zeros(n)
    SIZE = 0.25  # Conservative sizing
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = 250  # Need 200 SMA + CRSI
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(crsi[i]) or np.isnan(chop[i]) or np.isnan(atr[i]) or atr[i] <= 0:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(upper_d[i]) or np.isnan(sma_200[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        crsi_val = crsi[i]
        chop_val = chop[i]
        
        # === TREND FILTER (1w HMA aligned) ===
        trend_bullish = close[i] > hma_1w_aligned[i] if not np.isnan(hma_1w_aligned[i]) else True
        trend_bearish = close[i] < hma_1w_aligned[i] if not np.isnan(hma_1w_aligned[i]) else False
        
        # === REGIME DETECTION (Choppiness) ===
        is_choppy = chop_val > 61.8  # Range market - use CRSI mean reversion
        is_trending = chop_val < 38.2  # Trending market - use Donchian
        
        # === DONCHIAN BREAKOUT ===
        price_above_donch = close[i] > upper_d[i]
        price_below_donch = close[i] < lower_d[i]
        
        # === CRSI SIGNALS ===
        crsi_oversold = crsi_val < 10
        crsi_overbought = crsi_val > 90
        
        desired_signal = 0.0
        
        if not in_position:
            # === NEW ENTRY ===
            
            # Case 1: Choppy market - CRSI mean reversion
            if is_choppy:
                # Long on CRSI oversold + bullish 1w trend
                if crsi_oversold and trend_bullish:
                    desired_signal = SIZE
                
                # Short on CRSI overbought + bearish 1w trend
                if crsi_overbought and trend_bearish:
                    desired_signal = -SIZE
            
            # Case 2: Trending market - Donchian breakout
            elif is_trending:
                # Long on breakout above + trend aligned
                if price_above_donch and trend_bullish:
                    desired_signal = SIZE
                
                # Short on breakdown below + trend aligned
                if price_below_donch and trend_bearish:
                    desired_signal = -SIZE
            
            # Case 3: Neutral chop - require both CRSI extreme AND volume
            else:
                vol_spike = vol_ratio[i] > 1.5
                if crsi_oversold and vol_spike and trend_bullish:
                    desired_signal = SIZE
                if crsi_overbought and vol_spike and trend_bearish:
                    desired_signal = -SIZE
        
        # === STOPLOSS (2 ATR) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.0 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.0 * entry_atr
            stop_price = min(stop_price, trailing_stop)
            if high[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === TAKE PROFIT (3:1 ratio) ===
        tp_triggered = False
        
        if in_position and position_side > 0:
            profit_target = entry_price + 3.0 * entry_atr
            if high[i] >= profit_target:
                tp_triggered = True
        
        if in_position and position_side < 0:
            profit_target = entry_price - 3.0 * entry_atr
            if low[i] <= profit_target:
                tp_triggered = True
        
        if tp_triggered:
            desired_signal = 0.0  # Take profit and exit
        
        # === CRSI EXIT ===
        crsi_exit = False
        
        if in_position and position_side > 0:
            # Exit long on CRSI overbought
            if crsi_val > 85:
                crsi_exit = True
        
        if in_position and position_side < 0:
            # Exit short on CRSI oversold
            if crsi_val < 15:
                crsi_exit = True
        
        if crsi_exit:
            desired_signal = 0.0
        
        # === DONCHIAN EXIT ===
        donch_exit = False
        
        if in_position and position_side > 0:
            # Exit if price drops below lower Donchian
            if price_below_donch:
                donch_exit = True
        
        if in_position and position_side < 0:
            # Exit if price rises above upper Donchian
            if price_above_donch:
                donch_exit = True
        
        if donch_exit:
            desired_signal = 0.0
        
        # === UPDATE POSITION ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                if position_side > 0:
                    stop_price = entry_price - 2.0 * entry_atr
                else:
                    stop_price = entry_price + 2.0 * entry_atr
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