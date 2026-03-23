#!/usr/bin/env python3
"""
Experiment #447: 1d Primary + 4h HTF — Dual Regime with Connors RSI + Donchian Breakout

Hypothesis: Daily timeframe with 4h HTF bias can capture major swings while avoiding
whipsaw. Key improvements over #443 (Sharpe=0.133):
1. Use 4h HMA for intraday trend bias (not 1w which is too slow for 1d entries)
2. Connors RSI with relaxed thresholds (25/75) for mean reversion in chop
3. Donchian(20) breakout for trend following with 4h HMA confirmation
4. Choppiness Index regime filter (61.8/38.2 standard levels)
5. ATR trailing stop (2.5x) for risk management
6. Discrete position sizing: 0.0, ±0.25, ±0.30 to minimize fee churn

Target: Sharpe > 0.612, 80-200 trades over 4-year train, DD < -35%
Timeframe: 1d (proven for swing trading, 20-50 trades/year target)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_dual_regime_crsi_donchian_4h_v1"
timeframe = "1d"
leverage = 1.0

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

def calculate_rsi(close, period):
    """Calculate RSI using Wilder's smoothing."""
    n = len(close)
    rsi = np.full(n, np.nan)
    
    delta = np.diff(close)
    delta = np.insert(delta, 0, 0)
    
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    gain_s = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    loss_s = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = gain_s / (loss_s + 1e-10)
        rsi = 100.0 - (100.0 / (1.0 + rs))
    
    rsi = np.clip(rsi, 0, 100)
    return rsi

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_choppiness(high, low, close, period=14):
    """Calculate Choppiness Index (CHOP)."""
    n = len(close)
    chop = np.full(n, np.nan)
    
    atr = calculate_atr(high, low, close, period)
    
    for i in range(period, n):
        sum_atr = np.nansum(atr[i-period+1:i+1])
        highest_high = np.nanmax(high[i-period+1:i+1])
        lowest_low = np.nanmin(low[i-period+1:i+1])
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10 and sum_atr > 0:
            chop[i] = 100.0 * np.log10(sum_atr / price_range) / np.log10(period)
        else:
            chop[i] = 50.0
    
    chop = np.clip(chop, 0, 100)
    return chop

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """Calculate Connors RSI (CRSI)."""
    n = len(close)
    
    # RSI(3) component
    rsi = calculate_rsi(close, rsi_period)
    
    # Streak RSI component
    streak_rsi = np.full(n, np.nan)
    for i in range(streak_period, n):
        up_streak = 0
        down_streak = 0
        
        for j in range(i, max(i - streak_period - 5, 0), -1):
            if j == 0:
                break
            if close[j] > close[j-1]:
                up_streak += 1
                down_streak = 0
            elif close[j] < close[j-1]:
                down_streak += 1
                up_streak = 0
            else:
                break
        
        streak = up_streak if up_streak > 0 else -down_streak
        
        if streak > 0:
            streak_rsi[i] = 50.0 + (streak / (streak_period + 1)) * 50.0
        elif streak < 0:
            streak_rsi[i] = 50.0 - (abs(streak) / (streak_period + 1)) * 50.0
        else:
            streak_rsi[i] = 50.0
    
    streak_rsi = np.clip(streak_rsi, 0, 100)
    
    # Percent Rank component
    pct_rank = np.full(n, np.nan)
    returns = np.diff(close) / (close[:-1] + 1e-10)
    returns = np.insert(returns, 0, 0)
    
    for i in range(rank_period, n):
        window = returns[i-rank_period+1:i+1]
        current = returns[i]
        pct_rank[i] = 100.0 * np.sum(window < current) / rank_period
    
    with np.errstate(invalid='ignore'):
        crsi = (rsi + streak_rsi + pct_rank) / 3.0
    
    return crsi

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel."""
    n = len(high)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        upper[i] = np.nanmax(high[i-period+1:i+1])
        lower[i] = np.nanmin(low[i-period+1:i+1])
    
    return upper, lower

def calculate_sma(close, period):
    """Calculate Simple Moving Average."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    return sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate 1d indicators (primary timeframe)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    hma_21 = calculate_hma(close, 21)
    hma_50 = calculate_hma(close, 50)
    chop = calculate_choppiness(high, low, close, period=14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    atr_14 = calculate_atr(high, low, close, period=14)
    sma_200 = calculate_sma(close, 200)
    
    # Calculate and align 4h HMA for bias
    hma_4h_raw = calculate_hma(df_4h['close'].values, 21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate median ATR for vol filter
    valid_atr = atr_14[200:]
    atr_median = np.nanmedian(valid_atr[~np.isnan(valid_atr)])
    if np.isnan(atr_median) or atr_median <= 0:
        atr_median = np.nanmean(valid_atr[~np.isnan(valid_atr)])
    
    signals = np.zeros(n)
    BASE_SIZE = 0.25  # 25% base position size for 1d
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(250, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            continue
        if np.isnan(crsi[i]) or np.isnan(chop[i]):
            continue
        if np.isnan(hma_21[i]) or np.isnan(hma_50[i]):
            continue
        if np.isnan(hma_4h_aligned[i]):
            continue
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            continue
        if np.isnan(sma_200[i]):
            continue
        
        # === REGIME DETECTION (Choppiness Index) ===
        regime_chop = chop[i] > 61.8  # Range market (standard CHOP level)
        regime_trend = chop[i] < 38.2  # Trending market (standard CHOP level)
        
        # === HTF TREND BIAS (4h HMA) ===
        price_above_hma_4h = close[i] > hma_4h_aligned[i]
        price_below_hma_4h = close[i] < hma_4h_aligned[i]
        
        # === PRIMARY TREND (1d HMA) ===
        hma_bullish = hma_21[i] > hma_50[i]
        hma_bearish = hma_21[i] < hma_50[i]
        price_above_sma200 = close[i] > sma_200[i]
        price_below_sma200 = close[i] < sma_200[i]
        
        # === CRSI SIGNALS (Mean Reversion) — RELAXED THRESHOLDS ===
        crsi_oversold = crsi[i] < 25.0
        crsi_overbought = crsi[i] > 75.0
        crsi_extreme_oversold = crsi[i] < 15.0
        crsi_extreme_overbought = crsi[i] > 85.0
        
        # === DONCHIAN BREAKOUT (Trend Follow) ===
        donchian_breakout_long = close[i] > donchian_upper[i-1] if not np.isnan(donchian_upper[i-1]) else False
        donchian_breakout_short = close[i] < donchian_lower[i-1] if not np.isnan(donchian_lower[i-1]) else False
        
        # === VOL FILTER ===
        vol_ratio = atr_14[i] / (atr_median + 1e-10)
        if vol_ratio > 2.5:
            position_size = BASE_SIZE * 0.5
        elif vol_ratio > 1.5:
            position_size = BASE_SIZE * 0.75
        else:
            position_size = BASE_SIZE
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        signal_strength = 0  # Count confluence factors
        
        # === REGIME 1: CHOPPY/RANGE (CHOP > 61.8) — MEAN REVERSION ===
        if regime_chop:
            # Long: CRSI oversold + price above SMA200 (bullish bias)
            if crsi_oversold and price_above_sma200:
                signal_strength = 1
                if crsi_extreme_oversold:
                    signal_strength = 2
                if price_above_hma_4h:
                    signal_strength += 1
                desired_signal = position_size * (0.8 + 0.2 * signal_strength / 3)
            
            # Short: CRSI overbought + price below SMA200 (bearish bias)
            if crsi_overbought and price_below_sma200:
                if desired_signal == 0:
                    signal_strength = 1
                    if crsi_extreme_overbought:
                        signal_strength = 2
                    if price_below_hma_4h:
                        signal_strength += 1
                    desired_signal = -position_size * (0.8 + 0.2 * signal_strength / 3)
        
        # === REGIME 2: TRENDING (CHOP < 38.2) — TREND FOLLOW ===
        elif regime_trend:
            # Long: Donchian breakout + 4h HMA confirmation
            if donchian_breakout_long and price_above_hma_4h:
                signal_strength = 1
                if hma_bullish:
                    signal_strength += 1
                if price_above_sma200:
                    signal_strength += 1
                desired_signal = position_size * (0.8 + 0.2 * min(signal_strength, 3) / 3)
            # Long: HMA bullish + 4h HMA confirmation
            elif hma_bullish and price_above_hma_4h:
                if desired_signal == 0:
                    signal_strength = 1
                    if price_above_sma200:
                        signal_strength += 1
                    desired_signal = position_size * 0.7 * (0.8 + 0.2 * signal_strength / 2)
            
            # Short: Donchian breakdown + 4h HMA confirmation
            if donchian_breakout_short and price_below_hma_4h:
                if desired_signal == 0:
                    signal_strength = 1
                    if hma_bearish:
                        signal_strength += 1
                    if price_below_sma200:
                        signal_strength += 1
                    desired_signal = -position_size * (0.8 + 0.2 * min(signal_strength, 3) / 3)
            # Short: HMA bearish + 4h HMA confirmation
            elif hma_bearish and price_below_hma_4h:
                if desired_signal == 0:
                    signal_strength = 1
                    if price_below_sma200:
                        signal_strength += 1
                    desired_signal = -position_size * 0.7 * (0.8 + 0.2 * signal_strength / 2)
        
        # === REGIME 3: TRANSITION (38.2-61.8) — ONLY EXTREME SIGNALS ===
        else:
            # Only extreme CRSI signals with SMA200 filter
            if crsi_extreme_oversold and price_above_sma200:
                desired_signal = position_size * 0.5
            elif crsi_extreme_overbought and price_below_sma200:
                desired_signal = -position_size * 0.5
            # Only strong Donchian breakouts with 4h confirmation
            elif donchian_breakout_long and price_above_hma_4h and hma_bullish:
                desired_signal = position_size * 0.5
            elif donchian_breakout_short and price_below_hma_4h and hma_bearish:
                desired_signal = -position_size * 0.5
        
        # === CAP SIGNAL TO MAX 0.30 ===
        if desired_signal > 0.30:
            desired_signal = 0.30
        elif desired_signal < -0.30:
            desired_signal = -0.30
        
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
        
        # === CRSI EXTREME EXIT (Take Profit) ===
        if in_position and position_side > 0 and crsi[i] > 85.0:
            desired_signal = 0.0
        
        if in_position and position_side < 0 and crsi[i] < 15.0:
            desired_signal = 0.0
        
        # === HOLD LOGIC — Maintain position if bias unchanged ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0 and (hma_bullish or price_above_hma_4h):
                desired_signal = position_size
            elif position_side < 0 and (hma_bearish or price_below_hma_4h):
                desired_signal = -position_size
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal != 0.0:
            if desired_signal > 0:
                if desired_signal >= 0.28:
                    desired_signal = 0.30
                elif desired_signal >= 0.22:
                    desired_signal = 0.25
                else:
                    desired_signal = 0.15
            else:
                if desired_signal <= -0.28:
                    desired_signal = -0.30
                elif desired_signal <= -0.22:
                    desired_signal = -0.25
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