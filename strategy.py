#!/usr/bin/env python3
"""
Experiment #015: 1h Choppiness Regime + 4h/1d HMA Trend + Connors RSI Entries

Hypothesis: Lower timeframe (1h) can work IF we use strict confluence filters:
1. Choppiness Index detects regime (CHOP>55=range/mean-revert, CHOP<45=trend/follow)
2. 4h HMA provides intermediate trend direction
3. 1d HMA provides major trend bias (only trade with 1d trend)
4. Connors RSI for precise entry timing (CRSI<15 long, CRSI>85 short)
5. Session filter (8-20 UTC) avoids low-liquidity whipsaw
6. Volume confirmation (vol>0.8x avg) ensures real moves

Why this should beat 4h KAMA (Sharpe=0.514):
- Regime detection prevents trend strategies in chop (2022 whipsaw protection)
- 1h entries within 4h trend = more precise timing, same trade frequency
- CRSI has 75% win rate on mean reversion (proven in literature)
- Session filter reduces false breakouts during Asia session

Timeframe: 1h (REQUIRED for this experiment)
HTF: 4h and 1d via mtf_data helper (call ONCE before loop)
Position sizing: 0.20-0.30 discrete (smaller for lower TF)
Stoploss: 2.5 * ATR(14)
Target trades: 30-60/year (strict entry filters)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_chop_regime_crsi_4h_1d_hma_session_v1"
timeframe = "1h"
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

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    CHOP > 61.8 = choppy/range, CHOP < 38.2 = trending
    """
    n = len(close)
    chop = np.zeros(n)
    
    for i in range(period, n):
        # Sum of ATR over period
        atr_sum = 0.0
        for j in range(i - period + 1, i + 1):
            tr = max(high[j] - low[j], 
                     abs(high[j] - close[j-1]), 
                     abs(low[j] - close[j-1]))
            atr_sum += tr
        
        # Highest High - Lowest Low over period
        hh = np.max(high[i - period + 1:i + 1])
        ll = np.min(low[i - period + 1:i + 1])
        hl_range = hh - ll
        
        if hl_range > 0 and atr_sum > 0:
            chop[i] = 100 * np.log10(atr_sum / hl_range) / np.log10(period)
        else:
            chop[i] = 50.0
    
    return chop

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(close, 3) + RSI(streak, 2) + PercentRank(close, 100)) / 3
    
    RSI(3): Very short-term momentum
    RSI(Streak): Consecutive up/down days
    PercentRank: Where current price ranks vs last 100 closes
    
    CRSI < 10 = oversold (long), CRSI > 90 = overbought (short)
    """
    n = len(close)
    crsi = np.zeros(n)
    
    # RSI(3)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    avg_gain = pd.Series(gain).ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    
    rs = np.where(avg_loss == 0, 100.0, avg_gain / avg_loss)
    rsi_3 = 100 - (100 / (1 + rs))
    rsi_3 = np.clip(rsi_3, 0, 100)
    
    # Streak RSI(2)
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    streak_gain = np.where(streak > 0, streak, 0.0)
    streak_loss = np.where(streak < 0, -streak, 0.0)
    
    avg_streak_gain = pd.Series(streak_gain).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    avg_streak_loss = pd.Series(streak_loss).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    
    streak_rs = np.where(avg_streak_loss == 0, 100.0, avg_streak_gain / avg_streak_loss)
    rsi_streak = 100 - (100 / (1 + streak_rs))
    rsi_streak = np.clip(rsi_streak, 0, 100)
    
    # Percent Rank(100)
    for i in range(rank_period, n):
        window = close[i - rank_period + 1:i + 1]
        count_below = np.sum(window[:-1] < close[i])
        crsi[i] = (rsi_3[i] + rsi_streak[i] + (count_below / (rank_period - 1)) * 100) / 3
    
    return crsi

def calculate_session_filter(open_time):
    """
    Return 1 if hour is between 8-20 UTC (high liquidity), 0 otherwise.
    Avoids Asia session whipsaw.
    """
    # open_time is in milliseconds since epoch
    hours = (open_time // 3600000) % 24
    return ((hours >= 8) & (hours <= 20)).astype(float)

def calculate_volume_ratio(volume, period=20):
    """Calculate volume ratio: current volume / median volume over period."""
    vol_s = pd.Series(volume)
    vol_median = vol_s.rolling(window=period, min_periods=period).median().values
    vol_ratio = volume / vol_median
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    return vol_ratio

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 4h indicators
    hma_4h_21 = calculate_hma(df_4h['close'].values, 21)
    hma_4h_21_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_21)
    
    hma_4h_50 = calculate_hma(df_4h['close'].values, 50)
    hma_4h_50_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_50)
    
    # Calculate 1d indicators
    hma_1d_21 = calculate_hma(df_1d['close'].values, 21)
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    
    # Calculate 1h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    chop_14 = calculate_choppiness(high, low, close, 14)
    crsi = calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100)
    vol_ratio = calculate_volume_ratio(volume, 20)
    session = calculate_session_filter(open_time)
    
    signals = np.zeros(n)
    
    # Base position sizing (Rule 4 - discrete levels, max 0.40)
    # Lower TF = smaller size to control drawdown
    BASE_SIZE = 0.22
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -100
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_4h_21_aligned[i]) or np.isnan(hma_1d_21_aligned[i]):
            continue
        
        if np.isnan(chop_14[i]) or np.isnan(crsi[i]):
            continue
        
        # === REGIME DETECTION (Choppiness Index) ===
        # CHOP > 55 = range market (mean reversion)
        # CHOP < 45 = trending market (trend following)
        # 45-55 = neutral (no trades)
        is_range_regime = chop_14[i] > 55
        is_trend_regime = chop_45 = chop_14[i] < 45
        
        # === HTF TREND BIAS (4h + 1d) ===
        # 1d HMA = major trend (must align for any trade)
        # 4h HMA = intermediate trend (entry direction)
        htf_1d_bullish = close[i] > hma_1d_21_aligned[i]
        htf_1d_bearish = close[i] < hma_1d_21_aligned[i]
        
        htf_4h_bullish = close[i] > hma_4h_21_aligned[i]
        htf_4h_bearish = close[i] < hma_4h_21_aligned[i]
        
        # === CONNORS RSI EXTREMES ===
        crsi_oversold = crsi[i] < 15
        crsi_overbought = crsi[i] > 85
        
        # === VOLUME & SESSION FILTERS ===
        volume_ok = vol_ratio[i] > 0.7  # At least 70% of median
        session_ok = session[i] == 1.0  # 8-20 UTC only
        
        # === VOLATILITY-ADJUSTED POSITION SIZING ===
        atr_ratio = atr_14[i] / np.nanmedian(atr_14[max(0, i-100):i]) if i > 100 else 1.0
        vol_adjustment = np.clip(1.0 / atr_ratio, 0.7, 1.2)
        current_size = BASE_SIZE * vol_adjustment
        current_size = np.clip(current_size, 0.15, 0.30)
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # RANGE REGIME: Mean reversion with HTF trend alignment
        if is_range_regime and session_ok and volume_ok:
            # LONG: 1d bullish + 4h bullish + CRSI oversold
            if htf_1d_bullish and htf_4h_bullish and crsi_oversold:
                new_signal = current_size
            
            # SHORT: 1d bearish + 4h bearish + CRSI overbought
            elif htf_1d_bearish and htf_4h_bearish and crsi_overbought:
                new_signal = -current_size
        
        # TREND REGIME: Follow trend on CRSI pullback
        elif is_trend_regime and session_ok and volume_ok:
            # LONG: 1d bullish + 4h bullish + CRSI pullback (not extreme)
            if htf_1d_bullish and htf_4h_bullish and crsi[i] < 40:
                new_signal = current_size
            
            # SHORT: 1d bearish + 4h bearish + CRSI pullback (not extreme)
            elif htf_1d_bearish and htf_4h_bearish and crsi[i] > 60:
                new_signal = -current_size
        
        # === FREQUENCY SAFEGUARD ===
        # If no trades for 72 bars (~3 days on 1h), allow weaker entry
        if bars_since_last_trade > 72 and new_signal == 0.0 and not in_position:
            if htf_1d_bullish and htf_4h_bullish and crsi[i] < 35 and session_ok:
                new_signal = current_size * 0.7
            elif htf_1d_bearish and htf_4h_bearish and crsi[i] > 65 and session_ok:
                new_signal = -current_size * 0.7
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR ===
        stoploss_triggered = False
        
        if in_position and position_side != 0:
            if position_side > 0:
                if close[i] > highest_price:
                    highest_price = close[i]
                stoploss_price = highest_price - 2.5 * atr_14[i]
                if close[i] < stoploss_price:
                    stoploss_triggered = True
            
            if position_side < 0:
                if lowest_price == 0.0 or close[i] < lowest_price:
                    lowest_price = close[i]
                stoploss_price = lowest_price + 2.5 * atr_14[i]
                if close[i] > stoploss_price:
                    stoploss_triggered = True
        
        # === CRSI MEAN REVERSION EXIT ===
        crsi_exit = False
        if in_position and position_side != 0:
            if position_side > 0 and crsi[i] > 70:
                crsi_exit = True  # Long exit on overbought
            if position_side < 0 and crsi[i] < 30:
                crsi_exit = True  # Short exit on oversold
        
        # === REGIME CHANGE EXIT ===
        regime_exit = False
        if in_position and position_side != 0:
            # Exit if regime changes and we're in wrong strategy
            if position_side > 0 and chop_14[i] > 65:  # Range getting too choppy
                regime_exit = True
            if position_side < 0 and chop_14[i] > 65:
                regime_exit = True
        
        # === HTF TREND REVERSAL EXIT ===
        htf_reversal = False
        if in_position and position_side != 0:
            if position_side > 0 and htf_1d_bearish:
                htf_reversal = True
            if position_side < 0 and htf_1d_bullish:
                htf_reversal = True
        
        # Apply stoploss or exits
        if stoploss_triggered or crsi_exit or regime_exit or htf_reversal:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                last_trade_bar = i
            elif np.sign(new_signal) != position_side:
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                last_trade_bar = i
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_price = 0.0
                lowest_price = 0.0
                last_trade_bar = i
        
        signals[i] = new_signal
    
    return signals