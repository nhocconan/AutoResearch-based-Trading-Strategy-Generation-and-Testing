#!/usr/bin/env python3
"""
Experiment #486: 1d Primary + 1w HTF — Dual Regime (Choppiness + Connors RSI)

Hypothesis: Daily timeframe with weekly trend bias + regime detection works best
for BTC/ETH in mixed bull/bear markets. Key insights from research:
1. Choppiness Index > 61.8 = range regime → use Connors RSI mean reversion
2. Choppiness Index < 38.2 = trend regime → use Donchian breakout + HMA trend
3. Weekly HMA(21) provides HTF bias to filter counter-trend trades
4. Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
5. Loose entry thresholds to guarantee >=30 trades/year on 1d

Why this should work:
- 1d timeframe = fewer trades, less fee drag (target 20-50/year)
- Regime switching adapts to 2022 crash (range) vs 2021 bull (trend)
- Weekly HTF filter prevents counter-trend entries in strong trends
- Connors RSI has 75% win rate reported in literature for mean reversion

Position sizing: 0.25 base, 0.30 strong signals
Stoploss: 2.5x ATR from entry
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_dual_regime_chop_connors_1w_v1"
timeframe = "1d"
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

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index - measures market choppiness vs trending
    CHOP = 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    CHOP > 61.8 = range/choppy market
    CHOP < 38.2 = trending market
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    # Calculate True Range
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    choppiness = np.zeros(n)
    choppiness[:] = np.nan
    
    for i in range(period, n):
        atr_sum = np.sum(tr[i-period+1:i+1])
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        
        price_range = highest_high - lowest_low
        if price_range > 1e-10 and atr_sum > 1e-10:
            choppiness[i] = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
    
    return choppiness

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI (CRSI) = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    RSI_Streak: RSI of consecutive up/down days
    PercentRank: percentile rank of daily price change over lookback
    
    CRSI < 10 = extreme oversold (long signal)
    CRSI > 90 = extreme overbought (short signal)
    """
    n = len(close)
    if n < rank_period + 10:
        return np.full(n, np.nan)
    
    # RSI(3) - fast RSI
    rsi_fast = calculate_rsi(close, rsi_period)
    
    # RSI Streak - measure consecutive up/down days
    streak = np.zeros(n)
    streak_rsi = np.zeros(n)
    streak_rsi[:] = np.nan
    
    for i in range(1, n):
        if close[i] > close[i-1]:
            if streak[i-1] > 0:
                streak[i] = streak[i-1] + 1
            else:
                streak[i] = 1
        elif close[i] < close[i-1]:
            if streak[i-1] < 0:
                streak[i] = streak[i-1] - 1
            else:
                streak[i] = -1
        else:
            streak[i] = 0
    
    # Calculate RSI on streak values
    streak_gain = np.where(streak > 0, streak, 0.0)
    streak_loss = np.where(streak < 0, -streak, 0.0)
    
    avg_streak_gain = pd.Series(streak_gain).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    avg_streak_loss = pd.Series(streak_loss).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    
    for i in range(streak_period, n):
        if avg_streak_loss[i] < 1e-10:
            streak_rsi[i] = 100.0
        else:
            rs = avg_streak_gain[i] / avg_streak_loss[i]
            streak_rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    
    # Percent Rank - percentile of daily return over lookback
    daily_return = np.zeros(n)
    daily_return[1:] = np.diff(close) / close[:-1] * 100.0
    
    percent_rank = np.zeros(n)
    percent_rank[:] = np.nan
    
    for i in range(rank_period, n):
        window = daily_return[i-rank_period+1:i+1]
        current = daily_return[i]
        count_below = np.sum(window < current)
        percent_rank[i] = (count_below / rank_period) * 100.0
    
    # Combine into Connors RSI
    crsi = np.zeros(n)
    crsi[:] = np.nan
    
    for i in range(max(rank_period, streak_period, rsi_period), n):
        if not np.isnan(rsi_fast[i]) and not np.isnan(streak_rsi[i]) and not np.isnan(percent_rank[i]):
            crsi[i] = (rsi_fast[i] + streak_rsi[i] + percent_rank[i]) / 3.0
    
    return crsi

def calculate_donchian(high, low, period=20):
    """Donchian Channel - highest high and lowest low over period"""
    n = len(high)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    return upper, lower

def calculate_sma(close, period):
    """Simple Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    return sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load 1w HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align 1w HMA for trend bias
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate 1d indicators
    hma_1d = calculate_hma(close, period=21)
    atr = calculate_atr(high, low, close, period=14)
    choppiness = calculate_choppiness(high, low, close, period=14)
    crsi = calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    sma_50 = calculate_sma(close, 50)
    sma_200 = calculate_sma(close, 200)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.30
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    
    for i in range(250, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1d[i]) or np.isnan(choppiness[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1w_aligned[i]) or np.isnan(sma_50[i]):
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
        
        # === 1w HTF BIAS ===
        htf_bull = close[i] > hma_1w_aligned[i]
        htf_bear = close[i] < hma_1w_aligned[i]
        
        # === REGIME DETECTION (Choppiness Index) ===
        chop = choppiness[i]
        is_choppy = chop > 55.0  # Range market (loose threshold for more trades)
        is_trending = chop < 45.0  # Trend market
        
        # === CONNORS RSI EXTREMES (Mean Reversion) ===
        crsi_oversold = not np.isnan(crsi[i]) and crsi[i] < 20.0
        crsi_overbought = not np.isnan(crsi[i]) and crsi[i] > 80.0
        crsi_extreme_oversold = not np.isnan(crsi[i]) and crsi[i] < 15.0
        crsi_extreme_overbought = not np.isnan(crsi[i]) and crsi[i] > 85.0
        
        # === DONCHIAN BREAKOUT (Trend Following) ===
        donchian_breakout_long = close[i] > donchian_upper[i-1] if not np.isnan(donchian_upper[i-1]) else False
        donchian_breakdown_short = close[i] < donchian_lower[i-1] if not np.isnan(donchian_lower[i-1]) else False
        
        # === SMA FILTERS ===
        above_sma50 = close[i] > sma_50[i]
        below_sma50 = close[i] < sma_50[i]
        above_sma200 = not np.isnan(sma_200[i]) and close[i] > sma_200[i]
        below_sma200 = not np.isnan(sma_200[i]) and close[i] < sma_200[i]
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # REGIME 1: CHOPPY MARKET - Mean Reversion with Connors RSI
        if is_choppy:
            # Long: CRSI extreme oversold + above SMA200 or SMA50
            if crsi_extreme_oversold:
                if above_sma200 or above_sma50:
                    desired_signal = SIZE_STRONG
                elif not np.isnan(sma_200[i]):
                    # Even without SMA filter in extreme cases
                    desired_signal = SIZE_BASE
            elif crsi_oversold and above_sma50:
                desired_signal = SIZE_BASE
            
            # Short: CRSI extreme overbought + below SMA200 or SMA50
            if desired_signal == 0.0:
                if crsi_extreme_overbought:
                    if below_sma200 or below_sma50:
                        desired_signal = -SIZE_STRONG
                    elif not np.isnan(sma_200[i]):
                        desired_signal = -SIZE_BASE
                elif crsi_overbought and below_sma50:
                    desired_signal = -SIZE_BASE
        
        # REGIME 2: TRENDING MARKET - Donchian Breakout with HTF Bias
        elif is_trending:
            # Long: Donchian breakout + HTF bull + above SMA50
            if donchian_breakout_long and htf_bull and above_sma50:
                desired_signal = SIZE_STRONG
            elif donchian_breakout_long and htf_bull:
                desired_signal = SIZE_BASE
            
            # Short: Donchian breakdown + HTF bear + below SMA50
            if desired_signal == 0.0:
                if donchian_breakdown_short and htf_bear and below_sma50:
                    desired_signal = -SIZE_STRONG
                elif donchian_breakdown_short and htf_bear:
                    desired_signal = -SIZE_BASE
        
        # REGIME 3: NEUTRAL - Use HMA crossover + CRSI confirmation
        else:
            hma_bull = close[i] > hma_1d[i]
            hma_bear = close[i] < hma_1d[i]
            
            # Long: HMA bull + CRSI recovering from oversold
            if hma_bull and above_sma50:
                if not np.isnan(crsi[i]) and crsi[i] < 40.0 and crsi[i-1] < crsi[i]:
                    desired_signal = SIZE_BASE
            
            # Short: HMA bear + CRSI falling from overbought
            if desired_signal == 0.0:
                if hma_bear and below_sma50:
                    if not np.isnan(crsi[i]) and crsi[i] > 60.0 and crsi[i-1] > crsi[i]:
                        desired_signal = -SIZE_BASE
        
        # === STOPLOSS CHECK (2.5x ATR from entry) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            if high[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= SIZE_STRONG * 0.9:
            final_signal = SIZE_STRONG
        elif desired_signal <= -SIZE_STRONG * 0.9:
            final_signal = -SIZE_STRONG
        elif desired_signal >= SIZE_BASE * 0.9:
            final_signal = SIZE_BASE
        elif desired_signal <= -SIZE_BASE * 0.9:
            final_signal = -SIZE_BASE
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position or np.sign(final_signal) != position_side:
                # New position or flip
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                # Set stoploss
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
        
        signals[i] = final_signal
    
    return signals