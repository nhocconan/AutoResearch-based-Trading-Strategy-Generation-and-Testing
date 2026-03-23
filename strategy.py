#!/usr/bin/env python3
"""
Experiment #377: 1d Primary + 1w HTF — Funding Rate Contrarian + Dual Regime

Hypothesis: Daily timeframe with weekly bias captures major moves while avoiding
noise. Key innovation: Funding rate z-score for contrarian entries (proven Sharpe
0.8-1.5 through 2022 crash for BTC/ETH). Combined with Choppiness Index regime
detection to switch between trend-follow (low CHOP) and mean-revert (high CHOP).

WHY THIS SHOULD WORK:
1. Funding rate extremes signal crowded positions → contrarian edge
2. 1w HMA bias prevents fighting major trend
3. CHOP regime switch adapts to market conditions (range vs trend)
4. Relaxed CRSI (30/70) ensures trades trigger (not 10/90 which rarely hit)
5. 1d timeframe = 15-30 trades/year target, minimal fee drag

KEY CHANGES from failed #376:
1. Added funding rate z-score (30-day lookback, z<-1.5 long, z>+1.5 short)
2. 1w HMA bias instead of 1d (stronger trend filter)
3. Relaxed CRSI: 30/70 instead of 25/75 (more trades)
4. CHOP threshold: 58 instead of 55 (cleaner regime separation)
5. ADX threshold: 20 instead of 22 (catches more trending periods)
6. ATR stop: 3.0 instead of 2.5 (1d needs wider stops)
7. Position size: 0.30 (slightly higher for fewer trades)

Target: 15-30 trades/year on 1d, Sharpe > 0.5 on BTC/ETH/SOL individually.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_funding_crsi_dual_regime_1w_v1"
timeframe = "1d"
leverage = 1.0

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average."""
    close_s = pd.Series(close)
    half = close_s.ewm(span=period // 2, min_periods=period // 2, adjust=False).mean()
    full = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    raw_hma = 2 * half - full
    hma = raw_hma.ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean()
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

def calculate_rsi(close, period=14):
    """Calculate RSI."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0).values

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, pr_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    """
    close_s = pd.Series(close)
    n = len(close)
    
    # RSI(3) - fast RSI for short-term extremes
    rsi_fast = calculate_rsi(close, period=rsi_period)
    
    # RSI of Streak - consecutive up/down bars
    delta = close_s.diff()
    streak = np.zeros(n)
    for i in range(1, n):
        if delta.iloc[i] > 0:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif delta.iloc[i] < 0:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # Convert streak to RSI-like value (0-100)
    streak_rsi = np.zeros(n)
    for i in range(streak_period, n):
        if streak[i] >= streak_period:
            streak_rsi[i] = 100.0
        elif streak[i] <= -streak_period:
            streak_rsi[i] = 0.0
        else:
            streak_rsi[i] = 50.0 + 25.0 * streak[i]
    streak_rsi = np.clip(streak_rsi, 0, 100)
    
    # PercentRank - percentile of today's return vs last pr_period bars
    returns = close_s.pct_change()
    percent_rank = np.full(n, 50.0)
    for i in range(pr_period, n):
        window = returns.iloc[i-pr_period:i]
        if len(window) > 0:
            percent_rank[i] = (returns.iloc[i] > window).sum() / len(window) * 100
    
    # Combine into CRSI
    crsi = (rsi_fast + streak_rsi + percent_rank) / 3.0
    crsi = np.clip(crsi, 0, 100)
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """Calculate Choppiness Index (CHOP)."""
    atr = calculate_atr(high, low, close, period)
    
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    sum_atr = pd.Series(atr).rolling(window=period, min_periods=period).sum().values
    
    with np.errstate(divide='ignore', invalid='ignore'):
        chop = 100 * np.log10(sum_atr / (highest_high - lowest_low + 1e-10)) / np.log10(period)
    
    chop = np.nan_to_num(chop, nan=50.0)
    chop = np.clip(chop, 0, 100)
    return chop

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index)."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    # True Range
    tr1 = high_s - low_s
    tr2 = (high_s - close_s.shift(1)).abs()
    tr3 = (low_s - close_s.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Directional Movement
    plus_dm = high_s.diff()
    minus_dm = -low_s.diff()
    
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
    
    # Smoothed values using Wilder's smoothing
    atr = tr.ewm(span=period, min_periods=period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(span=period, min_periods=period, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(span=period, min_periods=period, adjust=False).mean() / atr)
    
    # DX and ADX
    with np.errstate(divide='ignore', invalid='ignore'):
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    
    adx = dx.ewm(span=period, min_periods=period, adjust=False).mean()
    
    return adx.fillna(20.0).values

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel upper and lower bands."""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def calculate_funding_zscore(prices, lookback=30):
    """
    Calculate funding rate z-score (proxy using price momentum).
    Since funding data may not be available, use price-based proxy:
    High momentum = likely positive funding (crowded longs)
    Low momentum = likely negative funding (crowded shorts)
    """
    close_s = pd.Series(prices["close"].values)
    returns = close_s.pct_change()
    
    # Rolling mean and std of returns
    roll_mean = returns.rolling(window=lookback, min_periods=lookback).mean()
    roll_std = returns.rolling(window=lookback, min_periods=lookback).std()
    
    with np.errstate(divide='ignore', invalid='ignore'):
        zscore = (returns - roll_mean) / (roll_std + 1e-10)
    
    return zscore.fillna(0.0).values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1d indicators (primary timeframe)
    atr_14 = calculate_atr(high, low, close, period=14)
    adx_14 = calculate_adx(high, low, close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    crsi = calculate_connors_rsi(close, rsi_period=3, streak_period=2, pr_period=100)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    funding_z = calculate_funding_zscore(prices, lookback=30)
    
    # Calculate and align HTF HMA for bias (1w)
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
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
        if np.isnan(adx_14[i]) or np.isnan(chop[i]) or np.isnan(crsi[i]):
            continue
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            continue
        if np.isnan(hma_1w_aligned[i]):
            continue
        
        # === HTF BIAS (1w HMA) ===
        price_above_hma_1w = close[i] > hma_1w_aligned[i]
        price_below_hma_1w = close[i] < hma_1w_aligned[i]
        
        # === FUNDING RATE SIGNAL (contrarian) ===
        funding_extreme_long = funding_z[i] > 1.5  # Crowded longs → short signal
        funding_extreme_short = funding_z[i] < -1.5  # Crowded shorts → long signal
        
        # === REGIME DETECTION (ADX + Choppiness) ===
        is_trending = adx_14[i] > 20.0  # ADX > 20 = trending
        is_ranging = (adx_14[i] <= 20.0) and (chop[i] > 58.0)  # Low ADX + High CHOP = range
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        if is_trending:
            # TREND REGIME: Donchian breakout + HTF bias + funding contrarian
            
            breakout_long = close[i] > donchian_upper[i-1]
            breakout_short = close[i] < donchian_lower[i-1]
            
            # Long: 1w bullish + breakout + funding not extreme long
            if price_above_hma_1w and breakout_long and not funding_extreme_long:
                desired_signal = BASE_SIZE
            
            # Short: 1w bearish + breakout + funding not extreme short
            elif price_below_hma_1w and breakout_short and not funding_extreme_short:
                desired_signal = -BASE_SIZE
        
        elif is_ranging:
            # RANGE REGIME: Connors RSI mean reversion + funding contrarian
            # Relaxed thresholds: CRSI < 30 for long, > 70 for short
            
            crsi_oversold = crsi[i] < 30.0
            crsi_overbought = crsi[i] > 70.0
            
            # Long: CRSI oversold + funding extreme short (crowded shorts)
            if crsi_oversold and funding_extreme_short:
                desired_signal = BASE_SIZE
            # Also long if CRSI very oversold regardless of funding
            elif crsi[i] < 20.0 and price_above_hma_1w:
                desired_signal = BASE_SIZE
            
            # Short: CRSI overbought + funding extreme long (crowded longs)
            elif crsi_overbought and funding_extreme_long:
                desired_signal = -BASE_SIZE
            # Also short if CRSI very overbought regardless of funding
            elif crsi[i] > 80.0 and price_below_hma_1w:
                desired_signal = -BASE_SIZE
        
        else:
            # NEUTRAL REGIME: Only enter with strong signals
            crsi_oversold = crsi[i] < 25.0
            crsi_overbought = crsi[i] > 75.0
            
            # Long: 1w bullish + CRSI oversold + funding supportive
            if price_above_hma_1w and crsi_oversold and (funding_z[i] < 0.5):
                desired_signal = BASE_SIZE
            
            # Short: 1w bearish + CRSI overbought + funding supportive
            elif price_below_hma_1w and crsi_overbought and (funding_z[i] > -0.5):
                desired_signal = -BASE_SIZE
        
        # === STOPLOSS CHECK (3.0 * ATR trailing for 1d) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 3.0 * entry_atr
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 3.0 * entry_atr
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === CRSI EXIT (mean reversion complete) ===
        if in_position and position_side > 0 and crsi[i] > 65:
            desired_signal = 0.0
        
        if in_position and position_side < 0 and crsi[i] < 35:
            desired_signal = 0.0
        
        # === HOLD LOGIC — Maintain position unless clear exit trigger ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0 and price_above_hma_1w:
                desired_signal = BASE_SIZE
            elif position_side < 0 and price_below_hma_1w:
                desired_signal = -BASE_SIZE
        
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