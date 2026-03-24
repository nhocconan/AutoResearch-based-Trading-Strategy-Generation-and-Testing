#!/usr/bin/env python3
"""
Experiment #954: 1d Primary + 1w HTF — Regime-Adaptive Dual Strategy

Hypothesis: Daily timeframe with weekly bias captures multi-week trends while avoiding 
noise of lower TFs. Key innovation: REGIME-SWITCHING between trend-follow and mean-reversion
based on Choppiness Index. This adapts to market conditions automatically.

Why this should work:
1. 1w trend bias: Weekly close > open = bullish (simple but proven effective)
2. Choppiness Index (14): CHOP < 38.2 = trending → use Donchian breakouts
   CHOP > 61.8 = ranging → use Connors RSI mean reversion
3. Connors RSI: (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
   Extreme values (<20/>80) signal oversold/overbought in ranges
4. Donchian(20): Breakout entries in trending regimes with weekly confirmation
5. ATR(14) 2.5x trailing stop for risk management

Entry conditions (LOOSE to guarantee trades):
- TREND REGIME (CHOP<38.2): Long if weekly bull + price > Donchian upper
  Short if weekly bear + price < Donchian lower
- RANGE REGIME (CHOP>61.8): Long if CRSI < 20, Short if CRSI > 80
- NEUTRAL (38.2≤CHOP≤61.8): No trades (wait for clarity)

Target: Sharpe>0.45, trades>=30 train, trades>=5 test, DD>-40%
Timeframe: 1d
Size: 0.28 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_regime_crsi_donchian_1w_v1"
timeframe = "1d"
leverage = 1.0

def calculate_hma(close, period):
    """
    Hull Moving Average (HMA)
    HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = max(1, period // 2)
    sqrt_n = max(1, int(np.sqrt(period)))
    
    def wma(series, span):
        result = np.full(len(series), np.nan)
        weights = np.arange(1, span + 1, dtype=np.float64)
        for i in range(span - 1, len(series)):
            window = series[i - span + 1:i + 1].astype(np.float64)
            result[i] = np.sum(window * weights) / np.sum(weights)
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    diff = np.full(n, np.nan, dtype=np.float64)
    for i in range(period - 1, n):
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            diff[i] = 2.0 * wma_half[i] - wma_full[i]
    
    hma = wma(diff, sqrt_n)
    return hma

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP)
    Measures if market is trending (low CHOP) or ranging (high CHOP)
    CHOP = 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    
    CHOP > 61.8 = ranging (mean revert)
    CHOP < 38.2 = trending (trend follow)
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    chop = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period, n):
        # Calculate ATR for each bar in window
        atr_sum = 0.0
        for j in range(i - period + 1, i + 1):
            if j == 0:
                tr = high[j] - low[j]
            else:
                tr = max(high[j] - low[j], 
                        abs(high[j] - close[j-1]), 
                        abs(low[j] - close[j-1]))
            atr_sum += tr
        
        # Highest high and lowest low over period
        highest_high = np.max(high[i - period + 1:i + 1])
        lowest_low = np.min(low[i - period + 1:i + 1])
        range_val = highest_high - lowest_low
        
        if range_val > 1e-10 and atr_sum > 1e-10:
            chop[i] = 100.0 * np.log10(atr_sum / range_val) / np.log10(period)
        else:
            chop[i] = 50.0  # neutral
    
    return chop

def calculate_crsi(close, rsi_period=3, streak_period=2, pr_period=100):
    """
    Connors RSI (CRSI)
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    RSI(3): Very short-term momentum
    RSI_Streak(2): Consecutive up/down days
    PercentRank(100): Where current price ranks vs last 100 days
    
    CRSI < 20 = oversold (long)
    CRSI > 80 = overbought (short)
    """
    n = len(close)
    if n < pr_period:
        return np.full(n, np.nan)
    
    # RSI(3)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    avg_gain = pd.Series(gain).ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    
    rsi_short = np.full(n, np.nan)
    for i in range(rsi_period, n):
        if avg_loss[i] > 1e-10:
            rs = avg_gain[i] / avg_loss[i]
            rsi_short[i] = 100.0 - (100.0 / (1.0 + rs))
        else:
            rsi_short[i] = 100.0
    
    # RSI Streak (2) - consecutive up/down days
    streak = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    # Convert streak to RSI-like value (0-100)
    streak_rsi = np.full(n, np.nan)
    for i in range(streak_period, n):
        streak_sum = 0.0
        streak_count = 0
        for j in range(i - streak_period + 1, i + 1):
            if streak[j] > 0:
                streak_sum += streak[j]
                streak_count += 1
        if streak_count > 0:
            avg_streak = streak_sum / streak_count
            streak_rsi[i] = min(100.0, max(0.0, 50.0 + avg_streak * 10.0))
        else:
            streak_rsi[i] = 50.0
    
    # Percent Rank (100)
    pr = np.full(n, np.nan, dtype=np.float64)
    for i in range(pr_period, n):
        window = close[i - pr_period + 1:i + 1]
        count_lower = np.sum(window[:-1] < close[i])
        pr[i] = count_lower / (pr_period - 1) * 100.0
    
    # Combine into CRSI
    crsi = np.full(n, np.nan, dtype=np.float64)
    for i in range(pr_period, n):
        if not np.isnan(rsi_short[i]) and not np.isnan(streak_rsi[i]) and not np.isnan(pr[i]):
            crsi[i] = (rsi_short[i] + streak_rsi[i] + pr[i]) / 3.0
    
    return crsi

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
    """Donchian Channel - highest high and lowest low over period"""
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
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Weekly trend bias: close > open = bullish
    weekly_bias_raw = (df_1w['close'].values - df_1w['open'].values)
    weekly_bias_aligned = align_htf_to_ltf(prices, df_1w, weekly_bias_raw)
    
    # Daily indicators
    chop = calculate_choppiness(high, low, close, period=14)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, pr_period=100)
    atr = calculate_atr(high, low, close, period=14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    
    signals = np.zeros(n)
    SIZE = 0.28
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(chop[i]) or np.isnan(crsi[i]) or np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(weekly_bias_aligned[i]) or np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === REGIME DETECTION ===
        is_trending = chop[i] < 38.2
        is_ranging = chop[i] > 61.8
        
        # === WEEKLY BIAS ===
        weekly_bull = weekly_bias_aligned[i] > 0.0
        weekly_bear = weekly_bias_aligned[i] < 0.0
        
        # === ENTRY LOGIC (LOOSE TO GUARANTEE TRADES) ===
        desired_signal = 0.0
        
        # TREND REGIME: Donchian breakout with weekly bias
        if is_trending:
            if weekly_bull and close[i] > donchian_upper[i - 1]:
                desired_signal = SIZE
            elif weekly_bear and close[i] < donchian_lower[i - 1]:
                desired_signal = -SIZE
        
        # RANGE REGIME: Connors RSI mean reversion (LOOSE thresholds)
        elif is_ranging:
            if crsi[i] < 20:
                desired_signal = SIZE
            elif crsi[i] > 80:
                desired_signal = -SIZE
        
        # === STOPLOSS CHECK (2.5x ATR trailing) ===
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
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= SIZE * 0.9:
            final_signal = SIZE
        elif desired_signal <= -SIZE * 0.9:
            final_signal = -SIZE
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position or np.sign(final_signal) != position_side:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                if position_side > 0:
                    stop_price = entry_price - 2.5 * entry_atr
                else:
                    stop_price = entry_price + 2.5 * entry_atr
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                stop_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = final_signal
    
    return signals