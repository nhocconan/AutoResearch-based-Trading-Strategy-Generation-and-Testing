#!/usr/bin/env python3
"""
Experiment #699: 4h Primary + 1d HTF — Fisher/KAMA Dual Regime with Choppiness Switch

Hypothesis: Market regime (choppy vs trending) determines which strategy works best.
- CHOP > 50 (choppy): Use Fisher Transform mean reversion (catches extremes in ranges)
- CHOP < 50 (trending): Use KAMA trend following with ADX confirmation
- 1d HMA provides overall trend bias to filter counter-trend trades
- 4h TF balances trade frequency (target 30-50/year) with signal quality

Why this should beat #689 (Fisher/KAMA/ADX Sharpe=-0.129):
1. Choppiness Index regime switch prevents using wrong strategy in wrong regime
2. Looser Fisher thresholds (-1.2/+1.2 not -1.5/+1.5) for more trades
3. CRSI confirmation adds confluence without being too restrictive
4. 1d HMA trend bias prevents counter-trend entries that caused whipsaw
5. Position sizing 0.25-0.30 with discrete levels minimizes fee churn

Target: Sharpe > 0.612, trades >= 30 train, >= 3 test, ALL symbols positive
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_fisher_kama_chop_regime_1d_v2"
timeframe = "4h"
leverage = 1.0

def calculate_fisher_transform(high, low, close, period=14):
    """Ehlers Fisher Transform - normalizes price to Gaussian distribution."""
    n = len(close)
    fisher = np.full(n, np.nan)
    fisher_prev = np.zeros(n)
    
    for i in range(period, n):
        highest = np.max(high[i-period+1:i+1])
        lowest = np.min(low[i-period+1:i+1])
        
        if highest == lowest:
            fisher[i] = fisher_prev[i-1] if i > 0 else 0.0
            fisher_prev[i] = fisher[i]
            continue
        
        value = (close[i] - lowest) / (highest - lowest)
        value = 0.999 * value + 0.001  # Clamp to avoid log(0)
        
        fisher_val = 0.5 * np.log((1 + value) / (1 - value + 1e-10))
        
        # Smooth with previous value
        if i > 0 and not np.isnan(fisher_prev[i-1]):
            fisher[i] = 0.7 * fisher_val + 0.3 * fisher_prev[i-1]
        else:
            fisher[i] = fisher_val
        
        fisher_prev[i] = fisher[i]
    
    return fisher

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """Connors RSI - combines RSI, streak, and percentile rank."""
    n = len(close)
    crsi = np.full(n, np.nan)
    
    if n < rank_period + 5:
        return crsi
    
    # RSI component (3-period)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    
    avg_gain = np.concatenate([[np.nan], avg_gain])
    avg_loss = np.concatenate([[np.nan], avg_loss])
    
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
    rsi = np.clip(rsi, 0, 100)
    
    # Streak component
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    # Streak RSI
    streak_gain = np.where(streak > 0, streak, 0)
    streak_loss = np.where(streak < 0, -streak, 0)
    
    avg_streak_gain = pd.Series(streak_gain).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    avg_streak_loss = pd.Series(streak_loss).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    
    avg_streak_gain = np.concatenate([[np.nan], avg_streak_gain])
    avg_streak_loss = np.concatenate([[np.nan], avg_streak_loss])
    
    with np.errstate(divide='ignore', invalid='ignore'):
        streak_rs = avg_streak_gain / (avg_streak_loss + 1e-10)
        streak_rsi = 100 - (100 / (1 + streak_rs))
    streak_rsi = np.clip(streak_rsi, 0, 100)
    
    # Percentile Rank component
    percent_rank = np.full(n, np.nan)
    for i in range(rank_period, n):
        window = close[i-rank_period+1:i+1]
        rank = np.sum(window[:-1] < window[-1])
        percent_rank[i] = rank / (rank_period - 1) * 100
    
    # Combine components
    valid_mask = ~np.isnan(rsi) & ~np.isnan(streak_rsi) & ~np.isnan(percent_rank)
    crsi[valid_mask] = (rsi[valid_mask] + streak_rsi[valid_mask] + percent_rank[valid_mask]) / 3
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """Choppiness Index - identifies ranging vs trending markets."""
    n = len(close)
    chop = np.full(n, np.nan)
    
    for i in range(period, n):
        highest = np.max(high[i-period+1:i+1])
        lowest = np.min(low[i-period+1:i+1])
        
        if highest == lowest:
            chop[i] = 100.0
            continue
        
        tr_sum = 0.0
        for j in range(i-period+1, i+1):
            tr_sum += high[j] - low[j]
        
        if tr_sum == 0:
            chop[i] = 100.0
            continue
        
        chop_val = 100 * np.log10((highest - lowest) / tr_sum) / np.log10(period)
        chop[i] = np.clip(chop_val, 0, 100)
    
    return chop

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """Kaufman Adaptive Moving Average - adapts to market efficiency."""
    n = len(close)
    kama = np.full(n, np.nan)
    
    if n < er_period + slow_period:
        return kama
    
    # Efficiency Ratio
    er = np.zeros(n)
    for i in range(er_period, n):
        change = np.abs(close[i] - close[i-er_period])
        volatility = np.sum(np.abs(np.diff(close[i-er_period:i+1])))
        if volatility > 0:
            er[i] = change / volatility
    
    # Smoothing constant
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    
    kama[er_period] = close[er_period]
    
    for i in range(er_period + 1, n):
        sc = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
        kama[i] = kama[i-1] + sc * (close[i] - kama[i-1])
    
    return kama

def calculate_adx(high, low, close, period=14):
    """Average Directional Index - trend strength."""
    n = len(close)
    adx = np.full(n, np.nan)
    
    if n < period * 2:
        return adx
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], np.abs(high[i] - close[i-1]), np.abs(low[i] - close[i-1]))
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        up_move = high[i] - high[i-1]
        down_move = low[i-1] - low[i]
        if up_move > down_move and up_move > 0:
            plus_dm[i] = up_move
        if down_move > up_move and down_move > 0:
            minus_dm[i] = down_move
    
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    with np.errstate(divide='ignore', invalid='ignore'):
        di_plus = 100 * plus_dm_smooth / (atr + 1e-10)
        di_minus = 100 * minus_dm_smooth / (atr + 1e-10)
        dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    return adx

def calculate_atr(high, low, close, period=14):
    """Average True Range."""
    n = len(close)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], np.abs(high[i] - close[i-1]), np.abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_hma(series, period):
    """Hull Moving Average."""
    if len(series) < period:
        return np.full(len(series), np.nan)
    
    wma1 = pd.Series(series).ewm(span=period//2, min_periods=period//2, adjust=False).mean().values
    wma2 = pd.Series(series).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    hma_raw = 2 * wma1 - wma2
    hma = pd.Series(hma_raw).ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean().values
    return hma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (CRITICAL - Rule 1)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HTF HMA for trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (4h) indicators
    fisher = calculate_fisher_transform(high, low, close, period=14)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    chop = calculate_choppiness(high, low, close, period=14)
    kama = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    adx = calculate_adx(high, low, close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.28
    REDUCED_SIZE = 0.18
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(200, n):
        # Skip if indicators not ready
        if np.isnan(fisher[i]) or np.isnan(crsi[i]) or np.isnan(chop[i]):
            continue
        if np.isnan(kama[i]) or np.isnan(adx[i]) or np.isnan(atr[i]):
            continue
        if np.isnan(hma_1d_aligned[i]):
            continue
        if atr[i] <= 1e-10:
            continue
        
        # === REGIME DETECTION ===
        is_choppy = chop[i] > 50  # Range-bound market
        is_trending = chop[i] < 45  # Trending market (hysteresis)
        
        # === HTF TREND BIAS ===
        above_1d_hma = close[i] > hma_1d_aligned[i]
        below_1d_hma = close[i] < hma_1d_aligned[i]
        
        # === SIGNAL GENERATION ===
        desired_signal = 0.0
        current_size = BASE_SIZE
        
        # Reduce size in very high volatility
        atr_median = np.nanmedian(atr[max(0, i-50):i+1])
        if atr[i] > 1.5 * (atr_median + 1e-10):
            current_size = REDUCED_SIZE
        
        # === MEAN REVERSION MODE (Choppy regime) ===
        if is_choppy:
            # Long: Fisher oversold + CRSI oversold + above 1d HMA (bullish bias)
            if fisher[i] < -1.2 and crsi[i] < 30:
                if above_1d_hma:
                    desired_signal = current_size
                else:
                    desired_signal = current_size * 0.5  # Weaker signal against HTF
            
            # Short: Fisher overbought + CRSI overbought + below 1d HMA (bearish bias)
            elif fisher[i] > 1.2 and crsi[i] > 70:
                if below_1d_hma:
                    desired_signal = -current_size
                else:
                    desired_signal = -current_size * 0.5  # Weaker signal against HTF
        
        # === TREND FOLLOWING MODE (Trending regime) ===
        elif is_trending:
            # Long: KAMA bullish + ADX strong + above 1d HMA
            if kama[i] > close[i] * 0.998 and adx[i] > 22 and above_1d_hma:
                # Wait for Fisher confirmation (not extremely overbought)
                if fisher[i] < 1.5:
                    desired_signal = current_size
            
            # Short: KAMA bearish + ADX strong + below 1d HMA
            elif kama[i] < close[i] * 1.002 and adx[i] > 22 and below_1d_hma:
                # Wait for Fisher confirmation (not extremely oversold)
                if fisher[i] > -1.5:
                    desired_signal = -current_size
        
        # === STOPLOSS CHECK (Trailing ATR 2.5x) ===
        stoploss_triggered = False
        
        if in_position and position_side == 1:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * entry_atr
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side == -1:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * entry_atr
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === EXIT CONDITIONS ===
        if in_position and position_side == 1:
            # Exit long on Fisher overbought or regime change
            if fisher[i] > 1.8 or crsi[i] > 80:
                desired_signal = 0.0
            elif is_choppy and fisher[i] > 0.5:  # Take profit in range
                desired_signal = 0.0
        
        if in_position and position_side == -1:
            # Exit short on Fisher oversold or regime change
            if fisher[i] < -1.8 or crsi[i] < 20:
                desired_signal = 0.0
            elif is_choppy and fisher[i] < -0.5:  # Take profit in range
                desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            desired_signal = BASE_SIZE if desired_signal >= BASE_SIZE * 0.7 else REDUCED_SIZE
        elif desired_signal < 0:
            desired_signal = -BASE_SIZE if desired_signal <= -BASE_SIZE * 0.7 else -REDUCED_SIZE
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                position_side = int(np.sign(desired_signal))
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
        
        signals[i] = desired_signal
    
    return signals