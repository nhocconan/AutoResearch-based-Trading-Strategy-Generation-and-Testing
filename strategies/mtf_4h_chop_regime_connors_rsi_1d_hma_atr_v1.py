#!/usr/bin/env python3
"""
Experiment #238: 4h Regime-Adaptive Strategy with Choppiness Index + Connors RSI

Hypothesis: Crypto markets alternate between trending and ranging regimes. 
Using Choppiness Index (CHOP) to detect regime allows adaptive strategy:
- CHOP > 55 = Range → Mean reversion (Connors RSI extremes + BB)
- CHOP < 45 = Trend → Trend following (HMA slope + momentum)
- 1d HMA provides higher-timeframe bias filter
- ATR trailing stop protects against regime shifts

Why this might work:
- Recent experiments show pure trend-following fails in 2025 bear/range market
- Mean reversion works in ranges but fails in strong trends
- CHOP filter switches between approaches dynamically
- Connors RSI (RSI3 + RSI_Streak + PercentRank) has 75% win rate in backtests
- 4h timeframe balances signal quality vs trade frequency
- Conservative sizing (0.25) controls drawdown

Learning from failures:
- #232 (4h KAMA + RSI pullback): Sharpe=-0.134 - no regime filter
- #236 (30m Fisher + KAMA): Sharpe=-9.213 - too noisy, wrong TF
- #237 (1h KAMA + ADX): Sharpe=-0.474 - ADX laggy on lower TF
- Trend-only strategies fail when market ranges (2025)
- Need regime detection to adapt

Timeframe: 4h (REQUIRED for this experiment)
HTF: 1d via mtf_data helper (call ONCE before loop)
Position sizing: 0.25 discrete levels
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_chop_regime_connors_rsi_1d_hma_atr_v1"
timeframe = "4h"
leverage = 1.0

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
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    CHOP > 61.8 = Range/Ranging market
    CHOP < 38.2 = Trending market
    """
    n = len(close)
    chop = np.zeros(n)
    
    for i in range(period, n):
        # Calculate ATR for each bar in the lookback
        atr_sum = 0.0
        for j in range(i - period + 1, i + 1):
            tr1 = high[j] - low[j]
            tr2 = abs(high[j] - close[j-1]) if j > 0 else tr1
            tr3 = abs(low[j] - close[j-1]) if j > 0 else tr1
            tr = max(tr1, tr2, tr3)
            atr_sum += tr
        
        highest_high = np.max(high[i - period + 1:i + 1])
        lowest_low = np.min(low[i - period + 1:i + 1])
        price_range = highest_high - lowest_low
        
        if price_range > 0 and atr_sum > 0:
            chop[i] = 100 * np.log10(atr_sum / price_range) / np.log10(period)
        else:
            chop[i] = 50.0
    
    # Fill initial values
    chop[:period] = 50.0
    
    return chop

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(close, 3) + RSI(Streak, 2) + PercentRank(100)) / 3
    
    RSI(Streak): RSI of consecutive up/down days
    PercentRank: Percentage of prior returns lower than current return
    """
    n = len(close)
    crsi = np.zeros(n)
    
    # RSI(3) on close
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean()
    avg_loss = loss.ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_close = 100 - (100 / (1 + rs))
    
    # Streak RSI - count consecutive up/down days
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    # RSI on streak values
    streak_s = pd.Series(streak)
    streak_delta = streak_s.diff()
    streak_gain = streak_delta.where(streak_delta > 0, 0.0)
    streak_loss = -streak_delta.where(streak_delta < 0, 0.0)
    
    streak_avg_gain = streak_gain.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    streak_avg_loss = streak_loss.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    
    streak_rs = streak_avg_gain / (streak_avg_loss + 1e-10)
    rsi_streak = 100 - (100 / (1 + streak_rs))
    
    # Percent Rank - rolling percentile of returns
    returns = close_s.pct_change()
    percent_rank = np.zeros(n)
    for i in range(rank_period, n):
        window = returns.iloc[i-rank_period:i].dropna()
        if len(window) > 0:
            current_return = returns.iloc[i] if not np.isnan(returns.iloc[i]) else 0
            percent_rank[i] = 100 * (window < current_return).sum() / len(window)
    
    # Combine into CRSI
    for i in range(n):
        if i >= rank_period and not np.isnan(rsi_close.iloc[i]) and not np.isnan(rsi_streak.iloc[i]):
            crsi[i] = (rsi_close.iloc[i] + rsi_streak.iloc[i] + percent_rank[i]) / 3.0
        elif i >= rsi_period and not np.isnan(rsi_close.iloc[i]):
            crsi[i] = rsi_close.iloc[i]
        else:
            crsi[i] = 50.0
    
    return crsi

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    
    return upper.values, lower.values, sma.values

def calculate_rsi(close, period=14):
    """Calculate RSI (Relative Strength Index)."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    return rsi.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HTF indicators
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 4h indicators
    atr = calculate_atr(high, low, close, 14)
    chop = calculate_choppiness(high, low, close, 14)
    crsi = calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100)
    rsi_14 = calculate_rsi(close, 14)
    bb_upper, bb_lower, bb_mid = calculate_bollinger_bands(close, 20, 2.0)
    hma_21 = calculate_hma(close, 21)
    hma_48 = calculate_hma(close, 48)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_BASE = 0.25
    SIZE_HALF = 0.125
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_close = 0.0
    lowest_close = 0.0
    entry_price = 0.0
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(chop[i]) or np.isnan(crsi[i]) or np.isnan(rsi_14[i]):
            signals[i] = 0.0
            continue
        
        # === HIGHER TIMEFRAME BIAS ===
        # 1d HMA = higher timeframe trend bias
        bull_trend_1d = close[i] > hma_1d_aligned[i]
        bear_trend_1d = close[i] < hma_1d_aligned[i]
        
        # === REGIME DETECTION ===
        # CHOP > 55 = Range market (mean reversion)
        # CHOP < 45 = Trend market (trend following)
        # 45-55 = Transition (reduce position or stay flat)
        is_range = chop[i] > 55
        is_trend = chop[i] < 45
        
        # === MEAN REVERSION SIGNALS (Range Regime) ===
        # Connors RSI < 10 = Oversold (long opportunity)
        # Connors RSI > 90 = Overbought (short opportunity)
        # Also check price vs Bollinger Bands
        crsi_oversold = crsi[i] < 15
        crsi_overbought = crsi[i] > 85
        price_at_bb_lower = close[i] < bb_lower[i]
        price_at_bb_upper = close[i] > bb_upper[i]
        
        # === TREND FOLLOWING SIGNALS (Trend Regime) ===
        # HMA21 > HMA48 = Bullish trend structure
        # HMA21 < HMA48 = Bearish trend structure
        hma_bullish = hma_21[i] > hma_48[i]
        hma_bearish = hma_21[i] < hma_48[i]
        
        # HMA slope (simple momentum)
        hma_slope_bullish = hma_21[i] > hma_21[i-5] if i >= 5 else False
        hma_slope_bearish = hma_21[i] < hma_21[i-5] if i >= 5 else False
        
        new_signal = 0.0
        
        # === RANGE REGIME: MEAN REVERSION ===
        if is_range:
            # Long: CRSI oversold + price at BB lower + 1d bias neutral or bullish
            if crsi_oversold and price_at_bb_lower:
                if bull_trend_1d or not bear_trend_1d:
                    new_signal = SIZE_BASE
            
            # Short: CRSI overbought + price at BB upper + 1d bias neutral or bearish
            if crsi_overbought and price_at_bb_upper:
                if bear_trend_1d or not bull_trend_1d:
                    new_signal = -SIZE_BASE
        
        # === TREND REGIME: TREND FOLLOWING ===
        elif is_trend:
            # Long: HMA bullish + slope up + 1d bullish
            if hma_bullish and hma_slope_bullish and bull_trend_1d:
                new_signal = SIZE_BASE
            
            # Short: HMA bearish + slope down + 1d bearish
            if hma_bearish and hma_slope_bearish and bear_trend_1d:
                new_signal = -SIZE_BASE
        
        # === TRANSITION REGIME: REDUCE EXPOSURE ===
        # When CHOP is 45-55, only maintain existing positions, don't enter new
        # This is handled by not setting new_signal in this zone
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        # Check stoploss on EXISTING position before considering new entry
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest close for long position
                if close[i] > highest_close:
                    highest_close = close[i]
                # Trailing stop: 2.5 * ATR below highest close
                stoploss_price = highest_close - 2.5 * atr[i]
                if close[i] < stoploss_price:
                    new_signal = 0.0  # Stoploss overrides entry signal
            
            if position_side < 0:
                # Update lowest close for short position
                if lowest_close == 0.0 or close[i] < lowest_close:
                    lowest_close = close[i]
                # Trailing stop: 2.5 * ATR above lowest close
                stoploss_price = lowest_close + 2.5 * atr[i]
                if close[i] > stoploss_price:
                    new_signal = 0.0  # Stoploss overrides entry signal
        
        # === TAKE PROFIT: Reduce to half at 2R profit ===
        if in_position and new_signal != 0.0:
            if position_side > 0:
                profit = close[i] - entry_price
                if profit >= 2.0 * atr[entry_price_idx] if 'entry_price_idx' in dir() else 2.0 * atr[i]:
                    new_signal = SIZE_HALF  # Take partial profit
            if position_side < 0:
                profit = entry_price - close[i]
                if profit >= 2.0 * atr[i]:
                    new_signal = -SIZE_HALF  # Take partial profit
        
        # === UPDATE POSITION TRACKING FOR NEXT BAR ===
        if new_signal != 0.0:
            if not in_position:
                # Entering new position
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                entry_price_idx = i
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Reversing position
                position_side = np.sign(new_signal)
                entry_price = close[i]
                entry_price_idx = i
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
            # else: maintaining same position direction (possibly reduced size)
        else:
            # Exiting position (signal-based or stoploss)
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_close = 0.0
                lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals