#!/usr/bin/env python3
"""
Experiment #474: 1d Primary + 1w HTF — Weekly Trend + Daily Regime Switch

Hypothesis: Daily timeframe with weekly trend filter should capture major multi-week
trends while avoiding the noise of lower timeframes. Historical data shows 1d strategies
have better risk-adjusted returns than 4h/6h because they filter out intraday whipsaw.

Key Design:
1. WEEKLY HMA(21) as primary trend filter - only trade in direction of weekly trend
2. DAILY REGIME: Choppiness Index (CHOP) to switch between trend-follow and mean-revert
3. TREND ENTRY: Daily HMA(16/48) crossover aligned with weekly trend
4. MEAN REVERT ENTRY: RSI(14) extremes (30/70) when CHOP > 61.8 (choppy regime)
5. DONCHIAN(20) breakout as alternative trend entry (catches strong moves)
6. ATR(14) stoploss at 2.5x for capital preservation
7. Position size: 0.25 base, 0.30 strong signals

Why this should work:
- Weekly filter prevents counter-trend trades (major source of losses in 2022)
- CHOP regime switch adapts to market conditions (trend vs range)
- Daily TF = ~30-50 trades/year = low fee drag
- Loose entry conditions ensure >=10 trades/symbol on train

Timeframe: 1d (daily)
HTF: 1w (weekly) - loaded ONCE before loop via mtf_data
Target: Sharpe>0.45, DD>-35%, trades>=120 train (30/year), trades>=15 test
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_weekly_trend_daily_regime_chop_v1"
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
    Choppiness Index (CHOP) - measures market choppiness vs trending
    CHOP > 61.8 = choppy/ranging market (mean reversion works)
    CHOP < 38.2 = trending market (trend following works)
    Formula: 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    # Calculate True Range
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    chop = np.zeros(n)
    chop[:] = np.nan
    
    for i in range(period, n):
        tr_sum = np.sum(tr[i-period+1:i+1])
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10 and tr_sum > 1e-10:
            chop[i] = 100.0 * np.log10(tr_sum / price_range) / np.log10(period)
    
    return chop

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
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align weekly HMA for major trend bias
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate daily indicators
    hma_1d_fast = calculate_hma(close, period=16)
    hma_1d_slow = calculate_hma(close, period=48)
    atr = calculate_atr(high, low, close, period=14)
    rsi = calculate_rsi(close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
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
        
        if np.isnan(hma_1d_fast[i]) or np.isnan(hma_1d_slow[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1w_aligned[i]) or np.isnan(chop[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(rsi[i]) or np.isnan(sma_50[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === WEEKLY TREND FILTER (major direction) ===
        weekly_bull = close[i] > hma_1w_aligned[i]
        weekly_bear = close[i] < hma_1w_aligned[i]
        
        # === DAILY REGIME (Choppiness Index) ===
        # CHOP > 61.8 = choppy (mean reversion)
        # CHOP < 38.2 = trending (trend following)
        # 38.2-61.8 = neutral (use weekly trend)
        choppy_regime = chop[i] > 61.8
        trending_regime = chop[i] < 38.2
        
        # === DAILY HMA CROSSOVER ===
        hma_cross_long = False
        hma_cross_short = False
        if i > 0:
            if not np.isnan(hma_1d_fast[i]) and not np.isnan(hma_1d_fast[i-1]):
                if not np.isnan(hma_1d_slow[i]) and not np.isnan(hma_1d_slow[i-1]):
                    if hma_1d_fast[i-1] <= hma_1d_slow[i-1] and hma_1d_fast[i] > hma_1d_slow[i]:
                        hma_cross_long = True
                    if hma_1d_fast[i-1] >= hma_1d_slow[i-1] and hma_1d_fast[i] < hma_1d_slow[i]:
                        hma_cross_short = True
        
        # === DONCHIAN BREAKOUT ===
        donchian_breakout_long = False
        donchian_breakdown_short = False
        if not np.isnan(donchian_upper[i-1]) and not np.isnan(donchian_lower[i-1]):
            donchian_breakout_long = close[i] > donchian_upper[i-1]
            donchian_breakdown_short = close[i] < donchian_lower[i-1]
        
        # === RSI EXTREMES ===
        rsi_oversold = rsi[i] < 35.0
        rsi_overbought = rsi[i] > 65.0
        
        # === SMA FILTER ===
        above_sma50 = close[i] > sma_50[i]
        above_sma200 = close[i] > sma_200[i] if not np.isnan(sma_200[i]) else True
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # REGIME 1: TRENDING (CHOP < 38.2) - follow weekly trend
        if trending_regime:
            # Long: Weekly bull + (HMA cross long OR Donchian breakout)
            if weekly_bull:
                if hma_cross_long or donchian_breakout_long:
                    desired_signal = SIZE_STRONG
                elif hma_1d_fast[i] > hma_1d_slow[i] and above_sma50:
                    # HMA already bullish + above SMA50
                    desired_signal = SIZE_BASE
            
            # Short: Weekly bear + (HMA cross short OR Donchian breakdown)
            elif weekly_bear:
                if hma_cross_short or donchian_breakdown_short:
                    desired_signal = -SIZE_STRONG
                elif hma_1d_fast[i] < hma_1d_slow[i] and not above_sma50:
                    # HMA already bearish + below SMA50
                    desired_signal = -SIZE_BASE
        
        # REGIME 2: CHOPPY (CHOP > 61.8) - mean reversion
        elif choppy_regime:
            # Long: RSI oversold + above SMA200 (long-term support)
            if rsi_oversold and above_sma200:
                desired_signal = SIZE_BASE
            
            # Short: RSI overbought + below SMA200 (long-term resistance)
            elif rsi_overbought and not above_sma200:
                desired_signal = -SIZE_BASE
        
        # REGIME 3: NEUTRAL (38.2 <= CHOP <= 61.8) - use weekly trend only
        else:
            # Long: Weekly bull + HMA fast > slow
            if weekly_bull and hma_1d_fast[i] > hma_1d_slow[i]:
                desired_signal = SIZE_BASE
            
            # Short: Weekly bear + HMA fast < slow
            elif weekly_bear and hma_1d_fast[i] < hma_1d_slow[i]:
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