#!/usr/bin/env python3
"""
Experiment #1529: 4h Primary + 1d HTF — Mean Reversion with Trend Filter

Hypothesis: Based on research showing BTC/ETH fail simple trend strategies but excel
at mean reversion in bear/range markets (2022 crash, 2025 bear market).

Key insights from 1100+ failed strategies:
- Pure trend following fails on BTC/ETH (negative Sharpe consistently)
- Mean reversion with HTF trend filter works better in bear markets
- Connors RSI has 75% win rate for reversals
- Choppiness Index prevents mean reversion during strong trends (whipsaw protection)
- 4h timeframe naturally generates 20-50 trades/year target
- LOOSE entry conditions are CRITICAL (#1518, #1528 had 0 trades from over-filtering)

Design:
- 1d HMA(21) for macro trend bias (only trade with daily trend)
- 4h Choppiness Index(14) to detect range vs trend regimes
- 4h Connors RSI for mean reversion entries (CRSI < 20 long, > 80 short)
- 4h Bollinger Bands(20, 2.0) for additional mean reversion confirmation
- 4h ATR(14) 2.5x trailing stop for risk management
- Position size 0.30 (discrete levels to minimize fee churn)
- Target: 80-200 trades/train (4 years), 20-50 trades/test (15 months)

Timeframe: 4h (as required by experiment)
HTF: 1d (daily trend bias)
Position Size: 0.30 (discrete: 0.0, ±0.20, ±0.30)
Target: Sharpe > 0.618 (beat current best), DD < -30%
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_meanrev_crsi_chop_1d_hma_atr_v1"
timeframe = "4h"
leverage = 1.0

def calculate_hma(close, period=21):
    """
    Hull Moving Average - reduces lag while maintaining smoothness
    HMA = WMA(2*WMA(n/2) - WMA(n))
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    def wma(data, w_period):
        result = np.full(len(data), np.nan)
        weights = np.arange(1, w_period + 1, dtype=float)
        weights /= weights.sum()
        for i in range(w_period - 1, len(data)):
            if np.any(np.isnan(data[i - w_period + 1:i + 1])):
                continue
            result[i] = np.sum(data[i - w_period + 1:i + 1] * weights)
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    diff = np.full(n, np.nan)
    mask = ~np.isnan(wma_half) & ~np.isnan(wma_full)
    diff[mask] = 2.0 * wma_half[mask] - wma_full[mask]
    
    hma = wma(diff, sqrt_period)
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

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP)
    Measures market choppiness vs trending
    CHOP > 61.8 = range/choppy market (good for mean reversion)
    CHOP < 38.2 = trending market (avoid mean reversion)
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    chop = np.full(n, np.nan)
    
    for i in range(period, n):
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        
        if highest_high == lowest_low:
            chop[i] = 100.0
            continue
        
        atr_sum = 0.0
        for j in range(i-period+1, i+1):
            tr = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
            atr_sum += tr
        
        chop[i] = 100.0 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(period)
    
    return chop

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI (CRSI)
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    Extreme values (<10 or >90) signal mean reversion opportunities
    Using looser thresholds (<20, >80) to ensure more trades
    """
    n = len(close)
    if n < rank_period:
        return np.full(n, np.nan)
    
    # RSI(3)
    rsi_short = calculate_rsi(close, rsi_period)
    
    # Streak RSI - measure consecutive up/down days
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    # Convert streak to RSI-like value
    streak_rsi = np.full(n, np.nan)
    for i in range(streak_period, n):
        pos_streaks = np.sum(streak[i-streak_period+1:i+1] > 0)
        streak_rsi[i] = 100.0 * pos_streaks / streak_period if streak_period > 0 else 50.0
    
    # Percent Rank - where does current return rank vs last 100 periods
    percent_rank = np.full(n, np.nan)
    for i in range(rank_period, n):
        returns = np.diff(close[i-rank_period+1:i+1])
        current_return = returns[-1] if len(returns) > 0 else 0
        rank = np.sum(returns[:-1] < current_return) / max(len(returns) - 1, 1)
        percent_rank[i] = 100.0 * rank
    
    # Combine into CRSI
    crsi = np.full(n, np.nan)
    mask = ~np.isnan(rsi_short) & ~np.isnan(streak_rsi) & ~np.isnan(percent_rank)
    crsi[mask] = (rsi_short[mask] + streak_rsi[mask] + percent_rank[mask]) / 3.0
    
    return crsi

def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Bollinger Bands"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    
    return upper, sma, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 1d HMA for trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (4h) indicators
    rsi = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    crsi = calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100)
    bb_upper, bb_mid, bb_lower = calculate_bollinger_bands(close, period=20, std_mult=2.0)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.30  # Appropriate size for 4h (20-50 trades/year target)
    
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
        if np.isnan(rsi[i]) or np.isnan(chop[i]) or np.isnan(crsi[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_1d_aligned[i]) or np.isnan(bb_upper[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === MACRO TREND (1d HMA) - primary direction bias ===
        daily_bull = close[i] > hma_1d_aligned[i]
        daily_bear = close[i] < hma_1d_aligned[i]
        
        # === REGIME DETECTION (Choppiness Index) ===
        # CHOP > 50 = choppy/range (good for mean reversion)
        # CHOP < 50 = trending (be cautious with mean reversion)
        choppy_regime = chop[i] > 50.0
        
        # === CONNORS RSI (Mean Reversion Signal) - LOOSE THRESHOLDS ===
        crsi_oversold = crsi[i] < 25.0  # Loose threshold for more long trades
        crsi_overbought = crsi[i] > 75.0  # Loose threshold for more short trades
        
        # === BOLLINGER BANDS (Additional Mean Reversion) ===
        bb_oversold = close[i] < bb_lower[i]  # Price below lower band
        bb_overbought = close[i] > bb_upper[i]  # Price above upper band
        
        # === RSI CONFIRMATION ===
        rsi_oversold = rsi[i] < 40.0  # Loose oversold
        rsi_overbought = rsi[i] > 60.0  # Loose overbought
        
        # === DESIRED SIGNAL - MEAN REVERSION WITH TREND FILTER ===
        desired_signal = 0.0
        
        # LONG SIGNALS - Mean reversion in downtrend within bullish daily trend
        if daily_bull:
            # Primary: CRSI oversold + choppy regime (best mean reversion setup)
            if crsi_oversold and choppy_regime:
                desired_signal = BASE_SIZE
            # Secondary: CRSI oversold + BB oversold (double confirmation)
            elif crsi_oversold and bb_oversold:
                desired_signal = BASE_SIZE
            # Tertiary: RSI oversold + price near BB lower (loose for trades)
            elif rsi_oversold and close[i] < bb_mid:
                desired_signal = BASE_SIZE * 0.7
            # Fallback: Daily bull + CRSI moderately oversold (ensures trade frequency)
            elif daily_bull and crsi[i] < 40.0:
                desired_signal = BASE_SIZE * 0.5
        
        # SHORT SIGNALS - Mean reversion in uptrend within bearish daily trend
        elif daily_bear:
            # Primary: CRSI overbought + choppy regime (best mean reversion setup)
            if crsi_overbought and choppy_regime:
                desired_signal = -BASE_SIZE
            # Secondary: CRSI overbought + BB overbought (double confirmation)
            elif crsi_overbought and bb_overbought:
                desired_signal = -BASE_SIZE
            # Tertiary: RSI overbought + price near BB upper (loose for trades)
            elif rsi_overbought and close[i] > bb_mid:
                desired_signal = -BASE_SIZE * 0.7
            # Fallback: Daily bear + CRSI moderately overbought (ensures trade frequency)
            elif daily_bear and crsi[i] > 60.0:
                desired_signal = -BASE_SIZE * 0.5
        
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
        if desired_signal >= BASE_SIZE * 0.8:
            final_signal = BASE_SIZE
        elif desired_signal >= BASE_SIZE * 0.6:
            final_signal = BASE_SIZE * 0.7
        elif desired_signal >= BASE_SIZE * 0.4:
            final_signal = BASE_SIZE * 0.5
        elif desired_signal <= -BASE_SIZE * 0.8:
            final_signal = -BASE_SIZE
        elif desired_signal <= -BASE_SIZE * 0.6:
            final_signal = -BASE_SIZE * 0.7
        elif desired_signal <= -BASE_SIZE * 0.4:
            final_signal = -BASE_SIZE * 0.5
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