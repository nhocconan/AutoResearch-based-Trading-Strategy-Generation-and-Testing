#!/usr/bin/env python3
"""
Experiment #279: 1h Regime-Adaptive Strategy with 4h HMA Bias + 12h ADX Filter

Hypothesis: After 278 experiments, pure trend-following fails in bear/range markets (2025).
This strategy adapts to market regime:

1. REGIME DETECTION (12h ADX):
   - ADX < 20 = Range market → Mean reversion strategy
   - ADX >= 20 = Trend market → Trend pullback strategy

2. MEAN REVERSION MODE (when ADX < 20):
   - Long: Price < BB_lower + Connors RSI < 15 + above 4h HMA
   - Short: Price > BB_upper + Connors RSI > 85 + below 4h HMA
   - Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3

3. TREND MODE (when ADX >= 20):
   - Long: Price > 4h HMA + pullback to EMA(21) + RSI(14) > 40
   - Short: Price < 4h HMA + rally to EMA(21) + RSI(14) < 60

4. VOLATILITY FILTER:
   - BB Width percentile to confirm regime (narrow = range, wide = trend)

5. RISK MANAGEMENT:
   - 2.5 * ATR trailing stoploss
   - Position size: 0.25 base, 0.35 in strong signals, 0.15 in high vol

Timeframe: 1h (REQUIRED for this experiment)
HTF: 4h HMA for trend bias, 12h ADX for regime detection
Position sizing: 0.15-0.35 discrete levels
Stoploss: 2.5 * ATR(14) trailing

Why this might work:
- Adapts to market regime (most failed strategies used one approach only)
- Connors RSI has 75% win rate in research literature
- 4h HMA prevents counter-trend trades (critical for 2022 crash)
- 12h ADX is slower/more reliable than 1h ADX for regime detection
- Looser entry thresholds to ensure >=10 trades per symbol
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_regime_adaptive_4h_hma_12h_adx_conners_v1"
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

def calculate_rsi(close, period=14):
    """Calculate RSI using standard formula."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values
    return rsi

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, pr_period=100):
    """
    Calculate Connors RSI: (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    RSI_Streak: RSI of consecutive up/down days
    PercentRank: Percentage of recent closes below current close
    """
    n = len(close)
    crsi = np.zeros(n)
    crsi[:] = np.nan
    
    # RSI(3)
    rsi_short = calculate_rsi(close, rsi_period)
    
    # RSI Streak - measure consecutive up/down days
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
        streak_window = streak[max(0, i-streak_period):i+1]
        if len(streak_window) >= streak_period:
            # Simple normalization: map streak to 0-100
            max_streak = np.max(np.abs(streak_window))
            if max_streak > 0:
                streak_rsi[i] = 50 + (streak[i] / max_streak) * 50
            else:
                streak_rsi[i] = 50
        else:
            streak_rsi[i] = 50
    
    # Percent Rank - percentage of recent closes below current
    for i in range(pr_period, n):
        window = close[i-pr_period:i+1]
        count_below = np.sum(window[:-1] < close[i])
        crsi[i] = (rsi_short[i] + streak_rsi[i] + (count_below / pr_period) * 100) / 3
    
    return crsi

def calculate_bollinger_bands(close, period=20, std_dev=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    return upper.values, lower.values, sma.values

def calculate_bb_width(upper, lower, middle):
    """Calculate Bollinger Band Width as percentage of middle."""
    width = (upper - lower) / middle
    return width

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index)."""
    n = len(close)
    adx = np.zeros(n)
    adx[:] = np.nan
    
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Directional Movement
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    for i in range(1, n):
        plus_move = high[i] - high[i-1]
        minus_move = low[i-1] - low[i]
        if plus_move > minus_move and plus_move > 0:
            plus_dm[i] = plus_move
        if minus_move > plus_move and minus_move > 0:
            minus_dm[i] = minus_move
    
    # Smoothed values using Wilder's smoothing (EMA with span=period)
    tr_s = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean()
    plus_dm_s = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean()
    minus_dm_s = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean()
    
    # Directional Indicators
    plus_di = 100 * plus_dm_s / tr_s
    minus_di = 100 * minus_dm_s / tr_s
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    dx = dx.fillna(0)
    adx_s = dx.ewm(span=period, min_periods=period, adjust=False).mean()
    adx = adx_s.values
    
    return adx

def calculate_ema(close, period=21):
    """Calculate Exponential Moving Average."""
    close_s = pd.Series(close)
    ema = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    return ema.values

def calculate_percentile_rank(values, window=100):
    """Calculate rolling percentile rank."""
    n = len(values)
    pr = np.zeros(n)
    pr[:] = np.nan
    for i in range(window, n):
        window_vals = values[i-window:i]
        pr[i] = np.sum(window_vals < values[i]) / window * 100
    return pr

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    adx_12h = calculate_adx(df_12h['high'].values, df_12h['low'].values, df_12h['close'].values, 14)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    
    # Calculate 1h indicators
    atr = calculate_atr(high, low, close, 14)
    bb_upper, bb_lower, bb_middle = calculate_bollinger_bands(close, 20, 2.0)
    bb_width = calculate_bb_width(bb_upper, bb_lower, bb_middle)
    rsi_14 = calculate_rsi(close, 14)
    connors_rsi = calculate_connors_rsi(close, 3, 2, 100)
    ema_21 = calculate_ema(close, 21)
    
    # BB Width percentile for regime confirmation
    bb_width_pr = calculate_percentile_rank(bb_width, 100)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.35
    SIZE_WEAK = 0.15
    SIZE_MAX = 0.40
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_close = 0.0
    lowest_close = 0.0
    entry_price = 0.0
    entry_atr = 0.0
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(adx_12h_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(connors_rsi[i]):
            signals[i] = 0.0
            continue
        
        # === REGIME DETECTION (12h ADX) ===
        # ADX < 20 = Range market (mean reversion)
        # ADX >= 20 = Trend market (trend following)
        adx_value = adx_12h_aligned[i]
        is_range_market = adx_value < 20
        is_trend_market = adx_value >= 20
        
        # BB Width percentile confirmation
        bb_width_percentile = bb_width_pr[i] if not np.isnan(bb_width_pr[i]) else 50
        bb_confirms_range = bb_width_percentile < 40  # Narrow bands = range
        bb_confirms_trend = bb_width_percentile > 60  # Wide bands = trend
        
        # === HIGHER TIMEFRAME BIAS (4h HMA) ===
        bull_trend_4h = close[i] > hma_4h_aligned[i]
        bear_trend_4h = close[i] < hma_4h_aligned[i]
        
        # === VOLATILITY ADJUSTMENT ===
        atr_recent_avg = np.nanmean(atr[max(0, i-20):i+1])
        high_volatility = atr[i] > 1.5 * atr_recent_avg if not np.isnan(atr_recent_avg) else False
        
        # Determine position size based on volatility
        if high_volatility:
            position_size = SIZE_WEAK
        else:
            position_size = SIZE_BASE
        
        # === GENERATE SIGNAL BASED ON REGIME ===
        new_signal = 0.0
        
        if is_range_market or bb_confirms_range:
            # === MEAN REVERSION MODE ===
            # Long: Price below BB lower + Connors RSI < 15 + above 4h HMA (trend filter)
            # Short: Price above BB upper + Connors RSI > 85 + below 4h HMA (trend filter)
            
            # Looser thresholds to ensure >=10 trades per symbol
            long_mr = (
                close[i] < bb_lower[i] * 1.005 and  # Price at or below BB lower
                connors_rsi[i] < 25 and  # Connors RSI oversold (looser than 15)
                bull_trend_4h  # Above 4h HMA (only long in uptrend)
            )
            
            short_mr = (
                close[i] > bb_upper[i] * 0.995 and  # Price at or above BB upper
                connors_rsi[i] > 75 and  # Connors RSI overbought (looser than 85)
                bear_trend_4h  # Below 4h HMA (only short in downtrend)
            )
            
            if long_mr:
                new_signal = SIZE_STRONG if connors_rsi[i] < 15 else position_size
            
            if short_mr:
                new_signal = -SIZE_STRONG if connors_rsi[i] > 85 else -position_size
        
        else:
            # === TREND MODE ===
            # Long: Price > 4h HMA + pullback to EMA21 + RSI(14) > 40
            # Short: Price < 4h HMA + rally to EMA21 + RSI(14) < 60
            
            # Pullback detection: price near EMA21 (within 1%)
            near_ema_long = abs(close[i] - ema_21[i]) / ema_21[i] < 0.015 if not np.isnan(ema_21[i]) else False
            near_ema_short = abs(close[i] - ema_21[i]) / ema_21[i] < 0.015 if not np.isnan(ema_21[i]) else False
            
            long_trend = (
                bull_trend_4h and  # Above 4h HMA
                near_ema_long and  # Pullback to EMA21
                rsi_14[i] > 35 and  # RSI not oversold (looser than 40)
                rsi_14[i] < 70  # RSI not overbought
            )
            
            short_trend = (
                bear_trend_4h and  # Below 4h HMA
                near_ema_short and  # Rally to EMA21
                rsi_14[i] < 65 and  # RSI not overbought (looser than 60)
                rsi_14[i] > 30  # RSI not oversold
            )
            
            if long_trend:
                new_signal = position_size
            
            if short_trend:
                new_signal = -position_size
        
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
        
        # === TREND REVERSAL EXIT ===
        # Exit if HTF bias reverses against position
        if in_position and new_signal != 0.0:
            if position_side > 0 and bear_trend_4h:
                new_signal = 0.0  # 4h trend reversed against long
            if position_side < 0 and bull_trend_4h:
                new_signal = 0.0  # 4h trend reversed against short
        
        # === UPDATE POSITION TRACKING FOR NEXT BAR ===
        if new_signal != 0.0:
            if not in_position:
                # Entering new position
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                entry_atr = atr[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Reversing position
                position_side = np.sign(new_signal)
                entry_price = close[i]
                entry_atr = atr[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
        else:
            # Exiting position (signal-based or stoploss)
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                highest_close = 0.0
                lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals