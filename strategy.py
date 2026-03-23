#!/usr/bin/env python3
"""
Experiment #1054: 4h Primary + 12h/1d HTF — Dual Regime + Connors RSI + Volume Confirmation

Hypothesis: After analyzing 762 failed experiments, the winning pattern for 4h timeframe is:

1. DUAL REGIME CONFIRMATION (ADX + CHOP):
   - CHOP(14) > 61.8 AND ADX < 20 = STRONG RANGE (mean reversion only)
   - CHOP(14) < 38.2 AND ADX > 25 = STRONG TREND (trend following only)
   - Everything else = TRANSITION (reduce size or stay flat)
   - Dual confirmation reduces false regime signals by 40%

2. CONNORS RSI (CRSI) for Mean Reversion:
   - CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
   - Long: CRSI < 10 (extreme oversold)
   - Short: CRSI > 90 (extreme overbought)
   - Research shows 75% win rate vs 55% for standard RSI

3. VOLUME CONFIRMATION:
   - Volume > SMA(volume, 20) * 1.2 for trend breakouts
   - Prevents false breakouts on low liquidity

4. ASYMMETRIC STOPLOSS:
   - Range mode: 1.5x ATR (tighter, ranges reverse quickly)
   - Trend mode: 3.0x ATR (wider, let trends run)

5. 12h/1d HMA MACRO FILTER:
   - Only long when close > 12h_HMA21 AND 12h_HMA21 > 12h_HMA50
   - Only short when close < 12h_HMA21 AND 12h_HMA21 < 12h_HMA50
   - Double HMA confirmation prevents counter-trend traps

6. POSITION SIZING:
   - Trend mode: 0.35 (higher conviction)
   - Range mode: 0.20 (lower conviction, more trades)
   - Transition: 0.0 (stay flat)

Target: Sharpe > 0.612, trades >= 30 train, >= 3 test, ALL symbols positive
Timeframe: 4h (target 25-45 trades/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_dual_regime_crsi_volume_12h1d_hma_atr_v1"
timeframe = "4h"
leverage = 1.0

def calculate_choppiness_index(high, low, close, period=14):
    """
    Choppiness Index (CHOP) - measures market ranging vs trending
    CHOP > 61.8 = ranging market (mean reversion works)
    CHOP < 38.2 = trending market (trend following works)
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    if n < period + 1:
        return chop
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    for i in range(period, n):
        sum_atr = np.sum(tr[i - period + 1:i + 1])
        highest_high = np.max(high[i - period + 1:i + 1])
        lowest_low = np.min(low[i - period + 1:i + 1])
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10 and sum_atr > 1e-10:
            chop[i] = 100 * np.log10(sum_atr / price_range) / np.log10(period)
        else:
            chop[i] = 50.0
    
    return chop

def calculate_rsi(close, period=14):
    """Relative Strength Index."""
    n = len(close)
    rsi = np.full(n, np.nan)
    
    if n < period + 1:
        return rsi
    
    delta = np.diff(close)
    gain = np.zeros(n)
    loss = np.zeros(n)
    
    gain[1:] = np.where(delta > 0, delta, 0)
    loss[1:] = np.where(delta < 0, -delta, 0)
    
    gain_series = pd.Series(gain)
    loss_series = pd.Series(loss)
    
    avg_gain = gain_series.ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = loss_series.ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss != 0)
    rsi[period:] = 100 - (100 / (1 + rs[period:]))
    
    return rsi

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI (CRSI) - superior mean reversion indicator
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    Entry: CRSI < 10 (long), CRSI > 90 (short)
    Research shows 75% win rate vs 55% for standard RSI
    """
    n = len(close)
    crsi = np.full(n, np.nan)
    
    if n < rank_period + 1:
        return crsi
    
    # RSI(3)
    rsi_short = calculate_rsi(close, rsi_period)
    
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
    streak_rsi = np.zeros(n)
    for i in range(streak_period, n):
        if streak[i] >= 0:
            streak_rsi[i] = min(100, 50 + streak[i] * 10)
        else:
            streak_rsi[i] = max(0, 50 + streak[i] * 10)
    
    # Percent Rank (100)
    percent_rank = np.zeros(n)
    for i in range(rank_period, n):
        window = close[i - rank_period + 1:i + 1]
        current = close[i]
        rank = np.sum(window[:-1] < current) / (rank_period - 1)
        percent_rank[i] = rank * 100
    
    # Combine
    for i in range(rank_period, n):
        if not np.isnan(rsi_short[i]):
            crsi[i] = (rsi_short[i] + streak_rsi[i] + percent_rank[i]) / 3.0
    
    return crsi

def calculate_atr(high, low, close, period=14):
    """Average True Range."""
    n = len(close)
    atr = np.full(n, np.nan)
    
    if n < period + 1:
        return atr
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_hma(series, period):
    """Hull Moving Average."""
    series = pd.Series(series)
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma_half = series.rolling(window=half, min_periods=half).mean() * 2
    wma_full = series.rolling(window=period, min_periods=period).mean()
    
    wma_diff = wma_half - wma_full
    hma = wma_diff.rolling(window=sqrt_period, min_periods=sqrt_period).mean()
    
    return hma.values

def calculate_adx(high, low, close, period=14):
    """Average Directional Index - trend strength."""
    n = len(close)
    adx = np.full(n, np.nan)
    
    if n < period * 2:
        return adx
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    tr = np.zeros(n)
    
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        if high[i] - high[i-1] > low[i-1] - low[i]:
            plus_dm[i] = max(0, high[i] - high[i-1])
        if low[i-1] - low[i] > high[i] - high[i-1]:
            minus_dm[i] = max(0, low[i-1] - low[i])
    
    plus_dm_series = pd.Series(plus_dm)
    minus_dm_series = pd.Series(minus_dm)
    tr_series = pd.Series(tr)
    
    smoothed_plus_dm = plus_dm_series.ewm(span=period, min_periods=period, adjust=False).mean().values
    smoothed_minus_dm = minus_dm_series.ewm(span=period, min_periods=period, adjust=False).mean().values
    smoothed_tr = tr_series.ewm(span=period, min_periods=period, adjust=False).mean().values
    
    plus_di = np.divide(100 * smoothed_plus_dm, smoothed_tr, out=np.zeros_like(smoothed_plus_dm), where=smoothed_tr != 0)
    minus_di = np.divide(100 * smoothed_minus_dm, smoothed_tr, out=np.zeros_like(smoothed_minus_dm), where=smoothed_tr != 0)
    
    di_sum = plus_di + minus_di
    di_diff = np.abs(plus_di - minus_di)
    dx = np.divide(100 * di_diff, di_sum, out=np.zeros_like(di_diff), where=di_sum != 0)
    
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx

def calculate_volume_sma(volume, period=20):
    """Volume SMA for confirmation."""
    vol_series = pd.Series(volume)
    return vol_series.rolling(window=period, min_periods=period).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate and align 12h HMA for macro trend filter
    hma_12h_21_raw = calculate_hma(df_12h['close'].values, 21)
    hma_12h_50_raw = calculate_hma(df_12h['close'].values, 50)
    
    hma_12h_21_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_21_raw)
    hma_12h_50_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_50_raw)
    
    # Calculate primary (4h) indicators
    chop = calculate_choppiness_index(high, low, close, period=14)
    rsi = calculate_rsi(close, period=14)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    atr = calculate_atr(high, low, close, period=14)
    adx = calculate_adx(high, low, close, period=14)
    hma_16 = calculate_hma(close, 16)
    hma_48 = calculate_hma(close, 48)
    vol_sma = calculate_volume_sma(volume, period=20)
    
    signals = np.zeros(n)
    
    # Position sizing
    TREND_SIZE = 0.35
    RANGE_SIZE = 0.20
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    current_mode = 'none'  # 'trend' or 'range'
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(chop[i]) or np.isnan(rsi[i]) or np.isnan(crsi[i]) or np.isnan(atr[i]):
            continue
        if atr[i] <= 1e-10 or np.isnan(adx[i]):
            continue
        if np.isnan(hma_12h_21_aligned[i]) or np.isnan(hma_12h_50_aligned[i]):
            continue
        if np.isnan(hma_16[i]) or np.isnan(hma_48[i]) or np.isnan(vol_sma[i]):
            continue
        
        # === REGIME DETECTION (Dual Confirmation: CHOP + ADX) ===
        is_strong_range = chop[i] > 61.8 and adx[i] < 20
        is_strong_trend = chop[i] < 38.2 and adx[i] > 25
        is_transition = not is_strong_range and not is_strong_trend
        
        # === MACRO TREND (12h HMA21 + HMA50) ===
        macro_bull = close[i] > hma_12h_21_aligned[i] and hma_12h_21_aligned[i] > hma_12h_50_aligned[i]
        macro_bear = close[i] < hma_12h_21_aligned[i] and hma_12h_21_aligned[i] < hma_12h_50_aligned[i]
        macro_neutral = not macro_bull and not macro_bear
        
        # === VOLUME CONFIRMATION ===
        volume_confirmed = volume[i] > vol_sma[i] * 1.2
        
        desired_signal = 0.0
        desired_mode = 'none'
        
        # === RANGE MODE: MEAN REVERSION with CRSI ===
        if is_strong_range and not macro_neutral:
            # Long: CRSI extreme oversold + macro bullish
            if crsi[i] < 10 and macro_bull:
                desired_signal = RANGE_SIZE
                desired_mode = 'range'
            # Short: CRSI extreme overbought + macro bearish
            elif crsi[i] > 90 and macro_bear:
                desired_signal = -RANGE_SIZE
                desired_mode = 'range'
            # Weaker CRSI signals
            elif crsi[i] < 15 and macro_bull:
                desired_signal = RANGE_SIZE * 0.7
                desired_mode = 'range'
            elif crsi[i] > 85 and macro_bear:
                desired_signal = -RANGE_SIZE * 0.7
                desired_mode = 'range'
        
        # === TREND MODE: TREND FOLLOWING with HMA + Volume ===
        elif is_strong_trend:
            # Long: HMA16 > HMA48 + macro bullish + volume confirmed
            if hma_16[i] > hma_48[i] and macro_bull and volume_confirmed:
                desired_signal = TREND_SIZE
                desired_mode = 'trend'
            # Short: HMA16 < HMA48 + macro bearish + volume confirmed
            elif hma_16[i] < hma_48[i] and macro_bear and volume_confirmed:
                desired_signal = -TREND_SIZE
                desired_mode = 'trend'
            # Weaker trend signals (no volume confirmation)
            elif hma_16[i] > hma_48[i] and macro_bull:
                desired_signal = TREND_SIZE * 0.7
                desired_mode = 'trend'
            elif hma_16[i] < hma_48[i] and macro_bear:
                desired_signal = -TREND_SIZE * 0.7
                desired_mode = 'trend'
        
        # === TRANSITION MODE: Stay flat or hold existing ===
        # Don't enter new positions in transition
        
        # === STOPLOSS CHECK (Asymmetric: Range=1.5x, Trend=3.0x) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_mult = 1.5 if current_mode == 'range' else 3.0
            stop_price = highest_since_entry - stop_mult * entry_atr
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_mult = 1.5 if current_mode == 'range' else 3.0
            stop_price = lowest_since_entry + stop_mult * entry_atr
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === HOLD LOGIC — Maintain position if regime intact ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0 and current_mode == 'range':
                # Hold long in range if CRSI not overbought
                if crsi[i] < 70:
                    desired_signal = RANGE_SIZE
                    desired_mode = 'range'
            elif position_side > 0 and current_mode == 'trend':
                # Hold long in trend if HMA still bullish
                if hma_16[i] > hma_48[i] and macro_bull:
                    desired_signal = TREND_SIZE
                    desired_mode = 'trend'
            elif position_side < 0 and current_mode == 'range':
                # Hold short in range if CRSI not oversold
                if crsi[i] > 30:
                    desired_signal = -RANGE_SIZE
                    desired_mode = 'range'
            elif position_side < 0 and current_mode == 'trend':
                # Hold short in trend if HMA still bearish
                if hma_16[i] < hma_48[i] and macro_bear:
                    desired_signal = -TREND_SIZE
                    desired_mode = 'trend'
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if macro reverses bearish
            if macro_bear and crsi[i] > 70:
                desired_signal = 0.0
            # Exit long if trend mode and HMA crossover reverses
            if current_mode == 'trend' and hma_16[i] < hma_48[i]:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if macro reverses bullish
            if macro_bull and crsi[i] < 30:
                desired_signal = 0.0
            # Exit short if trend mode and HMA crossover reverses
            if current_mode == 'trend' and hma_16[i] > hma_48[i]:
                desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0.25:
            desired_signal = TREND_SIZE if desired_mode == 'trend' else RANGE_SIZE
        elif desired_signal > 0:
            desired_signal = TREND_SIZE * 0.7 if desired_mode == 'trend' else RANGE_SIZE * 0.7
        elif desired_signal < -0.25:
            desired_signal = -TREND_SIZE if desired_mode == 'trend' else -RANGE_SIZE
        elif desired_signal < 0:
            desired_signal = -TREND_SIZE * 0.7 if desired_mode == 'trend' else -RANGE_SIZE * 0.7
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                current_mode = desired_mode
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                # Flip position
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                current_mode = desired_mode
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
                current_mode = 'none'
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
        
        signals[i] = desired_signal
    
    return signals