#!/usr/bin/env python3
"""
Experiment #191: 6h Primary + 1w/1d HTF — Multi-Regime with Volume Confirmation

Hypothesis: 6h timeframe sits between 4h (too noisy) and 12h (too slow). By using
1w for major trend bias and 1d for intermediate trend, we can filter 6h entries
to only trade in direction of higher timeframes. This reduces whipsaws that
destroyed previous 6h attempts.

Key innovations vs failed 6h strategies:
1. Volume confirmation filter (20-bar volume SMA) - avoids low-liquidity traps
2. Dual regime detection (ADX + BB Width) - more robust than CHOP alone
3. Connors RSI for mean reversion (proven 75% win rate in literature)
4. Donchian breakout for trending regimes with volume spike confirm
5. Asymmetric sizing: 0.30 for HTF-aligned trades, 0.20 for counter-trend MR

Regime Detection:
- ADX(14) > 25 + BB Width < 40th percentile = TREND → Donchian breakout
- ADX(14) < 20 + BB Width > 60th percentile = RANGE → Connors RSI extremes
- Transition zone (ADX 20-25) = reduced size or flat

HTF Filters (1w + 1d):
- 1w HMA(21) = major trend bias (only long if price > 1w HMA)
- 1d HMA(50) = intermediate trend (strengthens signal when aligned)

Entry Logic:
- TREND LONG: Donchian(20) breakout + volume > 1.5x avg + 1w/1d bull
- TREND SHORT: Donchian(20) breakdown + volume > 1.5x avg + 1w/1d bear
- RANGE LONG: CRSI < 20 + price > 1d HMA + volume confirm
- RANGE SHORT: CRSI > 80 + price < 1d HMA + volume confirm

Position Sizing:
- 0.30 for HTF-aligned trend trades
- 0.20 for mean reversion trades
- 0.15 for transition zone trades
- Stoploss: 2.5x ATR trailing

Target: Sharpe > 0.40 (beat #183's 0.399), trades >= 30 train, >= 3 test
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_regime_adx_bb_crsi_vol_1w1d_v1"
timeframe = "6h"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average - faster response than EMA"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma1 = pd.Series(close).ewm(span=half, min_periods=half, adjust=False).mean().values
    wma2 = pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    diff = 2.0 * wma1 - wma2
    hma = pd.Series(diff).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean().values
    
    return hma

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    gain = np.concatenate([[0.0], gain])
    loss = np.concatenate([[0.0], loss])
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.zeros(n)
    rsi[:] = np.nan
    for i in range(period, n):
        if avg_loss[i] < 1e-10:
            rsi[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_adx(high, low, close, period=14):
    """
    Average Directional Index - measures trend strength
    ADX > 25 = trending, ADX < 20 = ranging
    """
    n = len(close)
    if n < period * 3:
        return np.full(n, np.nan)
    
    # True Range
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    # Directional Movement
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    for i in range(1, n):
        plus_move = high[i] - high[i-1]
        minus_move = low[i-1] - low[i]
        if plus_move > minus_move and plus_move > 0:
            plus_dm[i] = plus_move
        elif minus_move > plus_move and minus_move > 0:
            minus_dm[i] = minus_move
    
    # Smoothed values
    atr_smooth = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_di = 100.0 * pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values / atr_smooth
    minus_di = 100.0 * pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values / atr_smooth
    
    # DX and ADX
    dx = np.zeros(n)
    dx[:] = np.nan
    for i in range(period, n):
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 1e-10:
            dx[i] = 100.0 * abs(plus_di[i] - minus_di[i]) / di_sum
    
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    return adx

def calculate_bollinger_bands(close, period=20, std_dev=2.0):
    """Bollinger Bands with width calculation"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    width = (upper - lower) / sma * 100.0  # BB Width as percentage
    
    return upper, lower, width

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, pr_period=100):
    """
    Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    Proven 75% win rate for mean reversion entries
    """
    n = len(close)
    if n < pr_period:
        return np.full(n, np.nan)
    
    # RSI(3)
    rsi_short = calculate_rsi(close, rsi_period)
    
    # Streak calculation
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # Streak RSI
    streak_rsi = np.zeros(n)
    streak_rsi[:] = np.nan
    for i in range(streak_period, n):
        streak_vals = streak[max(0, i-streak_period):i+1]
        up_streak = np.sum(streak_vals > 0)
        down_streak = np.sum(streak_vals < 0)
        total = up_streak + down_streak
        if total > 0:
            streak_rsi[i] = 100.0 * up_streak / total
        else:
            streak_rsi[i] = 50.0
    
    # Percent Rank of returns
    returns = np.zeros(n)
    for i in range(1, n):
        returns[i] = (close[i] - close[i-1]) / close[i-1] if close[i-1] > 1e-10 else 0.0
    
    percent_rank = np.zeros(n)
    percent_rank[:] = np.nan
    for i in range(pr_period, n):
        window = returns[i-pr_period:i]
        if len(window) > 0 and not np.isnan(returns[i]):
            count_below = np.sum(window < returns[i])
            percent_rank[i] = 100.0 * count_below / len(window)
    
    # Combine into CRSI
    crsi = np.zeros(n)
    crsi[:] = np.nan
    for i in range(pr_period, n):
        if not np.isnan(rsi_short[i]) and not np.isnan(streak_rsi[i]) and not np.isnan(percent_rank[i]):
            crsi[i] = (rsi_short[i] + streak_rsi[i] + percent_rank[i]) / 3.0
    
    return crsi

def calculate_donchian(high, low, period=20):
    """Donchian Channel - highest high and lowest low over period"""
    n = len(high)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    upper = np.zeros(n)
    lower = np.zeros(n)
    upper[:] = np.nan
    lower[:] = np.nan
    
    for i in range(period-1, n):
        upper[i] = np.max(high[i-period+1:i+1])
        lower[i] = np.min(low[i-period+1:i+1])
    
    return upper, lower

def calculate_volume_sma(volume, period=20):
    """Volume Simple Moving Average"""
    n = len(volume)
    if n < period:
        return np.full(n, np.nan)
    
    vol_sma = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    return vol_sma

def calculate_bb_width_percentile(bb_width, lookback=100):
    """Calculate BB Width percentile rank over lookback period"""
    n = len(bb_width)
    percentile = np.zeros(n)
    percentile[:] = np.nan
    
    for i in range(lookback, n):
        if not np.isnan(bb_width[i]):
            window = bb_width[i-lookback:i]
            valid_window = window[~np.isnan(window)]
            if len(valid_window) > 0:
                count_below = np.sum(valid_window < bb_width[i])
                percentile[i] = 100.0 * count_below / len(valid_window)
    
    return percentile

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 1w HMA for major trend bias
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate and align 1d HMA for intermediate trend
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=50)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (6h) indicators
    hma_6h = calculate_hma(close, period=21)
    atr = calculate_atr(high, low, close, period=14)
    adx = calculate_adx(high, low, close, period=14)
    bb_upper, bb_lower, bb_width = calculate_bollinger_bands(close, period=20, std_dev=2.0)
    bb_width_pct = calculate_bb_width_percentile(bb_width, lookback=100)
    crsi = calculate_connors_rsi(close, rsi_period=3, streak_period=2, pr_period=100)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    vol_sma = calculate_volume_sma(volume, period=20)
    
    signals = np.zeros(n)
    
    # Position sizing levels (discrete to minimize fee churn)
    SIZE_TREND = 0.30  # 30% for HTF-aligned trend trades
    SIZE_MR = 0.20     # 20% for mean reversion trades
    SIZE_TRANSITION = 0.15  # 15% for transition zone
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(250, n):  # Start after all indicators are ready
        # Skip if critical indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_6h[i]) or np.isnan(adx[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_1w_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(vol_sma[i]) or vol_sma[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (1w + 1d HMA) ===
        htf_1w_bull = close[i] > hma_1w_aligned[i]
        htf_1w_bear = close[i] < hma_1w_aligned[i]
        htf_1d_bull = close[i] > hma_1d_aligned[i]
        htf_1d_bear = close[i] < hma_1d_aligned[i]
        
        # Strong HTF alignment (both 1w and 1d agree)
        htf_strong_bull = htf_1w_bull and htf_1d_bull
        htf_strong_bear = htf_1w_bear and htf_1d_bear
        
        # === REGIME DETECTION (ADX + BB Width) ===
        is_trending = adx[i] > 25.0 and (bb_width_pct[i] < 40.0 or np.isnan(bb_width_pct[i]))
        is_ranging = adx[i] < 20.0 and (bb_width_pct[i] > 60.0 or np.isnan(bb_width_pct[i]))
        is_transition = not is_trending and not is_ranging  # ADX 20-25 or mixed BB signals
        
        # === VOLUME CONFIRMATION ===
        volume_spike = volume[i] > 1.5 * vol_sma[i] if not np.isnan(vol_sma[i]) else False
        volume_normal = volume[i] > 0.8 * vol_sma[i] if not np.isnan(vol_sma[i]) else False
        
        # === DONCHIAN BREAKOUT ===
        breakout_long = close[i] > donchian_upper[i-1] if not np.isnan(donchian_upper[i-1]) else False
        breakout_short = close[i] < donchian_lower[i-1] if not np.isnan(donchian_lower[i-1]) else False
        
        # === CRSI VALUES (for ranging regime) ===
        crsi_extreme_low = False
        crsi_extreme_high = False
        if not np.isnan(crsi[i]):
            crsi_extreme_low = crsi[i] < 20.0  # Less strict than 15 to generate more trades
            crsi_extreme_high = crsi[i] > 80.0  # Less strict than 85
        
        # === 6h HMA TREND ===
        hma_6h_bull = close[i] > hma_6h[i]
        hma_6h_bear = close[i] < hma_6h[i]
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # REGIME 1: TRENDING (Donchian breakout with volume + HTF alignment)
        if is_trending:
            # Long: Donchian breakout + volume spike + HTF bull
            if breakout_long and volume_spike and htf_1w_bull:
                if htf_strong_bull:
                    desired_signal = SIZE_TREND
                elif hma_6h_bull:
                    desired_signal = SIZE_TREND * 0.8
            
            # Short: Donchian breakdown + volume spike + HTF bear
            elif breakout_short and volume_spike and htf_1w_bear:
                if htf_strong_bear:
                    desired_signal = -SIZE_TREND
                elif hma_6h_bear:
                    desired_signal = -SIZE_TREND * 0.8
        
        # REGIME 2: RANGING (Connors RSI mean reversion)
        elif is_ranging:
            # Long: CRSI extreme low + above 1d HMA + volume normal
            if crsi_extreme_low and htf_1d_bull and volume_normal:
                desired_signal = SIZE_MR
            
            # Short: CRSI extreme high + below 1d HMA + volume normal
            elif crsi_extreme_high and htf_1d_bear and volume_normal:
                desired_signal = -SIZE_MR
        
        # REGIME 3: TRANSITION (reduced size, require stronger confirmation)
        elif is_transition:
            # Only enter if ALL align strongly (triple confluence)
            if breakout_long and htf_strong_bull and volume_spike and hma_6h_bull:
                desired_signal = SIZE_TRANSITION
            elif breakout_short and htf_strong_bear and volume_spike and hma_6h_bear:
                desired_signal = -SIZE_TRANSITION
            # Mean reversion in transition (less strict)
            elif crsi_extreme_low and htf_1d_bull:
                desired_signal = SIZE_MR * 0.7
            elif crsi_extreme_high and htf_1d_bear:
                desired_signal = -SIZE_MR * 0.7
        
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
        if desired_signal >= SIZE_TREND * 0.9:
            final_signal = SIZE_TREND
        elif desired_signal <= -SIZE_TREND * 0.9:
            final_signal = -SIZE_TREND
        elif desired_signal >= SIZE_MR * 0.9:
            final_signal = SIZE_MR
        elif desired_signal <= -SIZE_MR * 0.9:
            final_signal = -SIZE_MR
        elif desired_signal >= SIZE_TRANSITION * 0.9:
            final_signal = SIZE_TRANSITION
        elif desired_signal <= -SIZE_TRANSITION * 0.9:
            final_signal = -SIZE_TRANSITION
        elif abs(desired_signal) >= SIZE_MR * 0.5:
            final_signal = np.sign(desired_signal) * SIZE_TRANSITION
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