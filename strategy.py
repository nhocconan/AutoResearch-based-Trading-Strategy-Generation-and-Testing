#!/usr/bin/env python3
"""
Experiment #403: 1d Primary + 1w HTF — Connors RSI + HMA + Choppiness Regime

Hypothesis: Daily timeframe with weekly trend bias + Connors RSI mean reversion
will capture major moves while avoiding whipsaw. 1d TF targets 15-30 trades/year
with minimal fee drag (~0.75-1.5%).

Key innovations:
1. Connors RSI (CRSI) for entry timing: (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
   - More responsive than standard RSI, catches reversals faster
   - Long when CRSI < 15, Short when CRSI > 85
2. Weekly HMA(21) for major trend bias (1w HTF)
3. Daily HMA(8/21) crossover for trend confirmation
4. Choppiness Index regime: >61.8 = mean revert, <38.2 = trend follow
5. Donchian(20) breakout confirmation for entry timing
6. ATR(14) position sizing: reduce size when vol spikes >2x median
7. Asymmetric stoploss: 3.0x ATR for longs, 2.5x for shorts (wider for 1d)

Why this should beat Sharpe=0.612:
- 1d TF = fewer trades = less fee drag than 4h strategies
- CRSI proven in research (75% win rate on reversals)
- Weekly bias filter avoids counter-trend trades in strong trends
- Different from #399 (which used 4h + RSI(7) + dynamic thresholds)
- Targets 20-40 trades over 4 years = sustainable

Target: Sharpe > 0.612, 15-30 trades/year, DD < -40%
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_crsi_hma_chop_donchian_1w_v1"
timeframe = "1d"
leverage = 1.0

def calculate_hma(close, period):
    """Calculate Hull Moving Average."""
    close_s = pd.Series(close)
    wma1 = close_s.ewm(span=period // 2, min_periods=period // 2, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    diff = 2.0 * wma1 - wma2
    hma = diff.ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean()
    return hma.values

def calculate_rsi(close, period=14):
    """Calculate RSI."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / (avg_loss + 1e-10)
    
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.values

def calculate_crsi(close):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    RSI_Streak: RSI of consecutive up/down day streak
    PercentRank: percentile rank of 1-day price change over last 100 days
    """
    n = len(close)
    crsi = np.zeros(n)
    
    # RSI(3)
    rsi_3 = calculate_rsi(close, period=3)
    
    # RSI Streak (2)
    streak = np.zeros(n)
    streak_rsi = np.zeros(n)
    
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # Calculate RSI on absolute streak values
    streak_abs = np.abs(streak)
    streak_s = pd.Series(streak_abs)
    streak_delta = streak_s.diff()
    streak_gain = streak_delta.where(streak_delta > 0, 0.0)
    streak_loss = -streak_delta.where(streak_delta < 0, 0.0)
    
    avg_streak_gain = streak_gain.ewm(span=2, min_periods=2, adjust=False).mean()
    avg_streak_loss = streak_loss.ewm(span=2, min_periods=2, adjust=False).mean()
    
    with np.errstate(divide='ignore', invalid='ignore'):
        streak_rs = avg_streak_gain / (avg_streak_loss + 1e-10)
    
    streak_rsi = 100.0 - (100.0 / (1.0 + streak_rs))
    streak_rsi = np.nan_to_num(streak_rsi, nan=50.0)
    
    # PercentRank(100) - percentile of 1-day return over last 100 days
    returns = np.diff(close) / (close[:-1] + 1e-10)
    percent_rank = np.zeros(n)
    
    for i in range(100, n):
        window = returns[i-100:i]
        current_return = returns[i-1]
        rank = np.sum(window < current_return) / len(window)
        percent_rank[i] = rank * 100.0
    
    # Combine into CRSI
    for i in range(100, n):
        crsi[i] = (rsi_3[i] + streak_rsi[i] + percent_rank[i]) / 3.0
    
    crsi[:100] = 50.0  # Default for warmup period
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """Calculate Choppiness Index (CHOP)."""
    n = len(close)
    chop = np.zeros(n)
    
    for i in range(period, n):
        highest = high[i-period+1:i+1].max()
        lowest = low[i-period+1:i+1].min()
        
        if highest - lowest < 1e-10:
            chop[i] = 50.0
            continue
        
        tr_sum = 0.0
        for j in range(i-period+1, i+1):
            tr1 = high[j] - low[j]
            tr2 = abs(high[j] - close[j-1]) if j > 0 else tr1
            tr3 = abs(low[j] - close[j-1]) if j > 0 else tr1
            tr = max(tr1, tr2, tr3)
            tr_sum += tr
        
        chop[i] = 100.0 * np.log10(tr_sum / (highest - lowest)) / np.log10(period)
        chop[i] = np.clip(chop[i], 0, 100)
    
    return chop

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (upper and lower)."""
    n = len(high)
    upper = np.zeros(n)
    lower = np.zeros(n)
    
    for i in range(period-1, n):
        upper[i] = high[i-period+1:i+1].max()
        lower[i] = low[i-period+1:i+1].min()
    
    return upper, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate daily indicators (primary timeframe)
    hma_8 = calculate_hma(close, 8)
    hma_21 = calculate_hma(close, 21)
    crsi = calculate_crsi(close)
    chop = calculate_choppiness(high, low, close, period=14)
    atr_14 = calculate_atr(high, low, close, period=14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    
    # Calculate and align HTF HMA for bias (1w)
    hma_1w_raw = calculate_hma(df_1w['close'].values, 21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate median ATR for vol filter
    atr_median = np.nanmedian(atr_14[150:])
    
    signals = np.zeros(n)
    BASE_SIZE = 0.30  # 30% position size for 1d (target 15-30 trades/year)
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(200, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            continue
        if np.isnan(chop[i]) or np.isnan(crsi[i]):
            continue
        if np.isnan(hma_1w_aligned[i]) or np.isnan(hma_8[i]) or np.isnan(hma_21[i]):
            continue
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            continue
        
        # === CHOPPINESS REGIME ===
        is_choppy = chop[i] > 61.8  # Range market - mean revert
        is_trending = chop[i] < 38.2  # Trend market - trend follow
        
        # === HTF BIAS (1w HMA) ===
        price_above_hma_1w = close[i] > hma_1w_aligned[i]
        price_below_hma_1w = close[i] < hma_1w_aligned[i]
        
        # === PRIMARY TREND (1d HMA crossover) ===
        hma_bullish = hma_8[i] > hma_21[i]
        hma_bearish = hma_8[i] < hma_21[i]
        
        # === CONNORS RSI EXTREMES ===
        crsi_oversold = crsi[i] < 15.0  # Strong buy signal
        crsi_overbought = crsi[i] > 85.0  # Strong sell signal
        crsi_neutral_low = crsi[i] < 30.0  # Moderate buy
        crsi_neutral_high = crsi[i] > 70.0  # Moderate sell
        
        # === DONCHIAN BREAKOUT ===
        donchian_long = close[i] > donchian_upper[i-1]
        donchian_short = close[i] < donchian_lower[i-1]
        
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
        
        # LONG SETUP
        long_bias = price_above_hma_1w  # Weekly trend bullish
        
        if long_bias:
            if is_choppy:
                # Mean reversion in range: CRSI oversold
                if crsi_oversold:
                    desired_signal = position_size
                elif crsi_neutral_low and hma_bullish:
                    desired_signal = position_size * 0.5
            elif is_trending:
                # Trend following: HMA bullish + CRSI not overbought
                if hma_bullish and crsi[i] < 70.0:
                    desired_signal = position_size
            else:
                # Neutral regime: combine signals
                if hma_bullish and (crsi_neutral_low or donchian_long):
                    desired_signal = position_size
        
        # SHORT SETUP
        short_bias = price_below_hma_1w  # Weekly trend bearish
        
        if short_bias:
            if is_choppy:
                # Mean reversion in range: CRSI overbought
                if crsi_overbought:
                    desired_signal = -position_size
                elif crsi_neutral_high and hma_bearish:
                    desired_signal = -position_size * 0.5
            elif is_trending:
                # Trend following: HMA bearish + CRSI not oversold
                if hma_bearish and crsi[i] > 30.0:
                    desired_signal = -position_size
            else:
                # Neutral regime: combine signals
                if hma_bearish and (crsi_neutral_high or donchian_short):
                    desired_signal = -position_size
        
        # === STOPLOSS CHECK (Asymmetric: wider for 1d) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 3.0 * entry_atr  # 3.0x for longs (1d needs wider)
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * entry_atr  # 2.5x for shorts
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === CRSI EXIT (extreme reached - take profit) ===
        if in_position and position_side > 0 and crsi_overbought:
            desired_signal = 0.0
        
        if in_position and position_side < 0 and crsi_oversold:
            desired_signal = 0.0
        
        # === TREND EXIT (HTF bias reversal) ===
        if in_position and position_side > 0 and price_below_hma_1w:
            desired_signal = 0.0
        
        if in_position and position_side < 0 and price_above_hma_1w:
            desired_signal = 0.0
        
        # === HMA EXIT (trend reversal on daily) ===
        if in_position and position_side > 0 and hma_bearish:
            desired_signal = 0.0
        
        if in_position and position_side < 0 and hma_bullish:
            desired_signal = 0.0
        
        # === HOLD LOGIC — Maintain position unless clear exit trigger ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0 and long_bias and hma_bullish:
                desired_signal = position_size
            elif position_side < 0 and short_bias and hma_bearish:
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
                # Position flip
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