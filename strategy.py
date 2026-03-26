#!/usr/bin/env python3
"""
Experiment #004: 1d KAMA Trend + CRSI Momentum + 1w Regime Filter

HYPOTHESIS: Daily KAMA captures smooth trend direction, filtering noise.
CRSI (Connors RSI) identifies mean-reversion entry points within the trend.
Weekly HMA confirms regime (bull/bear) to avoid counter-trend trades.
This combination generates fewer but higher-quality signals suitable for
the slow 1d timeframe (7-25 trades/year). Works in both directions by
checking both KAMA slope and CRSI extremes.

TIMEFRAME: 1d primary, 1w for regime
TARGET: 50-120 total trades over 4 years (12-30/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_kama_crsi_1w_regime_v1"
timeframe = "1d"
leverage = 1.0

def calculate_kama(close, period=21, fast_ema=2, slow_ema=30):
    """Kaufman's Adaptive Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    # Calculate Efficiency Ratio (ER)
    direction = np.abs(close[period-1:] - close[:n-(period-1)])
    volatility = np.zeros(n - period + 1)
    for i in range(n - period):
        volatility[i] = np.sum(np.abs(np.diff(close[i:i+period])))
    volatility = np.concatenate([[np.nan] * (period - 1), volatility])
    
    er = np.zeros(n)
    for i in range(period - 1, n):
        if volatility[i] > 1e-10:
            er[i] = direction[i] / volatility[i]
    
    # Calculate smoothing constant
    fast_const = (fast_ema / slow_ema) ** 2
    sc = (er * fast_const + (1 - er) * (2 / (slow_ema + 1)) ** 0.5) ** 2
    
    kama = np.full(n, np.nan, dtype=np.float64)
    kama[period - 1] = close[period - 1]
    
    for i in range(period, n):
        if not np.isnan(sc[i]) and not np.isnan(kama[i-1]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama

def calculate_crsi(close, period=14, streak_len=2, rank_period=100):
    """Connors RSI: (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3"""
    n = len(close)
    if n < rank_period:
        return np.full(n, 50.0)
    
    # RSI(3)
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(span=3, min_periods=3, adjust=False).mean()
    avg_loss = loss.ewm(span=3, min_periods=3, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi3 = (100 - (100 / (1 + rs)))
    
    # RSI Streak (2)
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = max(streak[i-1] + 1, 1)
        elif close[i] < close[i-1]:
            streak[i] = min(streak[i-1] - 1, -1)
        else:
            streak[i] = 0
    
    streak_series = pd.Series(streak)
    streak_delta = streak_series.diff()
    streak_gain = streak_delta.where(streak_delta > 0, 0.0)
    streak_loss = (-streak_delta).where(streak_delta < 0, 0.0)
    streak_avg_gain = streak_gain.ewm(span=2, min_periods=2, adjust=False).mean()
    streak_avg_loss = streak_loss.ewm(span=2, min_periods=2, adjust=False).mean()
    streak_rs = streak_avg_gain / (streak_avg_loss + 1e-10)
    rsi_streak = (100 - (100 / (1 + streak_rs)))
    
    # PercentRank(100)
    percent_rank = pd.Series(close).rolling(rank_period, min_periods=rank_period).apply(
        lambda x: (x[-1] < x).sum() / len(x) * 100, raw=True
    )
    
    # Combined CRSI
    crsi = (rsi3 + rsi_streak + percent_rank) / 3.0
    
    return crsi.values

def calculate_hma(close, period):
    """Hull Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = max(1, period // 2)
    sqrt_n = max(1, int(np.sqrt(period)))
    
    def wma(series, span):
        result = np.full(len(series), np.nan, dtype=np.float64)
        weights = np.arange(1, span + 1, dtype=np.float64)
        weight_sum = np.sum(weights)
        for i in range(span - 1, len(series)):
            if not np.isnan(series[i]):
                window = series[i - span + 1:i + 1].astype(np.float64)
                if not np.any(np.isnan(window)):
                    result[i] = np.sum(window * weights) / weight_sum
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    diff = np.full(n, np.nan, dtype=np.float64)
    for i in range(period - 1, n):
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            diff[i] = 2.0 * wma_half[i] - wma_full[i]
    
    return wma(diff, sqrt_n)

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

def calculate_donchian(high, low, period=20):
    """Donchian Channel - returns upper and lower bands"""
    n = len(high)
    upper = np.full(n, np.nan, dtype=np.float64)
    lower = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
    
    return upper, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE ===
    df_1w = get_htf_data(prices, '1w')
    
    # 1w HMA for regime
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate local 1d indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # KAMA 21
    kama = calculate_kama(close, period=21)
    
    # CRSI (3 components combined)
    crsi = calculate_crsi(close, period=14, streak_len=2, rank_period=100)
    
    # Volume MA
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # KAMA slope (trend direction)
    kama_slope = pd.Series(kama).diff(5).values
    
    # Donchian for structure
    donch_upper, donch_lower = calculate_donchian(high, low, period=20)
    
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
    last_entry_bar = -100  # Minimum bars between entries
    
    warmup = 150  # Need at least 100 for CRSI percentrank
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        
        if np.isnan(kama[i]) or np.isnan(crsi[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === 1w REGIME CHECK ===
        price_above_1w_hma = not np.isnan(hma_1w_aligned[i]) and close[i] > hma_1w_aligned[i]
        price_below_1w_hma = not np.isnan(hma_1w_aligned[i]) and close[i] < hma_1w_aligned[i]
        regime_bull = price_above_1w_hma
        regime_bear = price_below_1w_hma
        
        # === KAMA TREND ===
        kama_up = kama_slope[i] > 0 if not np.isnan(kama_slope[i]) else False
        kama_down = kama_slope[i] < 0 if not np.isnan(kama_slope[i]) else False
        
        # === CRSI MOMENTUM ===
        crsi_oversold = crsi[i] < 20  # Very oversold - reversal likely
        crsi_overbought = crsi[i] > 80  # Very overbought - reversal likely
        crsi_neutral_high = crsi[i] > 50
        crsi_neutral_low = crsi[i] < 50
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.2
        
        # === DONCHIAN STRUCTURE ===
        price_near_upper = close[i] > donch_upper[i] - atr_14[i] if not np.isnan(donch_upper[i]) else False
        price_near_lower = close[i] < donch_lower[i] + atr_14[i] if not np.isnan(donch_lower[i]) else False
        
        # Minimum bars since last entry
        bars_since_entry = i - last_entry_bar
        min_bars_passed = bars_since_entry > 5
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position and min_bars_passed:
            # === LONG ENTRY ===
            # CRSI oversold + KAMA turning up + volume + bull regime
            if crsi_oversold and (kama_up or crsi[i] > crsi[i-3]):  # CRSI bouncing or KAMA rising
                if vol_spike and regime_bull:
                    desired_signal = SIZE
            
            # Alternative: Strong momentum in bull regime
            if not desired_signal and regime_bull and crsi[i] < 30 and vol_spike:
                desired_signal = SIZE * 0.5  # Half size - less confident
            
            # === SHORT ENTRY ===
            # CRSI overbought + KAMA turning down + volume + bear regime
            if crsi_overbought and (kama_down or crsi[i] < crsi[i-3]):  # CRSI falling or KAMA declining
                if vol_spike and regime_bear:
                    desired_signal = -SIZE
            
            # Alternative: Strong momentum in bear regime
            if desired_signal == 0 and regime_bear and crsi[i] > 70 and vol_spike:
                desired_signal = -SIZE * 0.5  # Half size - less confident
        
        # === STOPLOSS CHECK (2.5 ATR) ===
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
        
        # === TAKE PROFIT (3:1) ===
        take_profit = False
        
        if in_position and position_side > 0:
            profit_target = entry_price + 3.0 * entry_atr
            if high[i] >= profit_target:
                take_profit = True
        
        if in_position and position_side < 0:
            profit_target = entry_price - 3.0 * entry_atr
            if low[i] <= profit_target:
                take_profit = True
        
        if take_profit:
            desired_signal = 0.0
        
        # === TRAILING STOP ADJUSTMENT ===
        if in_position and position_side > 0:
            # Trail stop when in profit
            if close[i] > entry_price + 1.5 * entry_atr:
                stop_price = max(stop_price, entry_price + 0.5 * entry_atr)
        
        if in_position and position_side < 0:
            if close[i] < entry_price - 1.5 * entry_atr:
                stop_price = min(stop_price, entry_price - 0.5 * entry_atr)
        
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
                last_entry_bar = i
                if position_side > 0:
                    stop_price = entry_price - 2.5 * entry_atr
                else:
                    stop_price = entry_price + 2.5 * entry_atr
        else:
            if in_position and (stoploss_triggered or take_profit or 
                               (i - last_entry_bar > 50)):  # Max hold ~50 days
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                stop_price = 0.0
        
        signals[i] = desired_signal if in_position else 0.0
    
    return signals