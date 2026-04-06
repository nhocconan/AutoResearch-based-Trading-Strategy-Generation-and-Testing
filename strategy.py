#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Daily candlestick reversal patterns (Hammer/Shooting Star) with volume confirmation and weekly trend filter on 12h timeframe
# Works in bull/bear because reversal patterns capture turning points, volume filters weak signals, and weekly trend ensures we trade with the dominant trend.
# Target: 60-150 trades over 4 years (15-38/year) to balance opportunity and fee cost.

name = "exp_12988_12h_daily_reversal_volume_trend_v1"
timeframe = "12h"
leverage = 1.0

# Parameters
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
BODY_RATIO = 0.3  # body size relative to total range
WICK_RATIO = 2.0   # wick size relative to body

def calculate_ema(close, period):
    """Calculate EMA with proper warmup"""
    return pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values

def calculate_engulfing_signal(open_prices, high, low, close):
    """Detect bullish/bearish engulfing patterns"""
    n = len(close)
    bullish_engulf = np.zeros(n, dtype=bool)
    bearish_engulf = np.zeros(n, dtype=bool)
    
    for i in range(1, n):
        # Bullish engulfing: current green candle fully engulfs previous red candle
        if (close[i] > open_prices[i] and  # current bullish
            close[i-1] < open_prices[i-1] and  # previous bearish
            open_prices[i] <= close[i-1] and  # current open <= previous close
            close[i] >= open_prices[i-1]):    # current close >= previous open
            bullish_engulf[i] = True
        
        # Bearish engulfing: current red candle fully engulfs previous green candle
        if (close[i] < open_prices[i] and  # current bearish
            close[i-1] > open_prices[i-1] and  # previous bullish
            open_prices[i] >= close[i-1] and  # current open >= previous close
            close[i] <= open_prices[i-1]):    # current close <= previous open
            bearish_engulf[i] = True
    
    return bullish_engulf, bearish_engulf

def calculate_hammer_shooting_star(open_prices, high, low, close):
    """Detect hammer and shooting star patterns"""
    n = len(close)
    hammer = np.zeros(n, dtype=bool)
    shooting_star = np.zeros(n, dtype=bool)
    
    for i in range(n):
        body_size = abs(close[i] - open_prices[i])
        total_range = high[i] - low[i]
        
        if total_range == 0:
            continue
            
        lower_wick = min(open_prices[i], close[i]) - low[i]
        upper_wick = high[i] - max(open_prices[i], close[i])
        
        # Hammer: small body, long lower wick, little/no upper wick
        if (body_size / total_range <= BODY_RATIO and
            lower_wick >= WICK_RATIO * body_size and
            upper_wick <= body_size):
            hammer[i] = True
            
        # Shooting star: small body, long upper wick, little/no lower wick
        if (body_size / total_range <= BODY_RATIO and
            upper_wick >= WICK_RATIO * body_size and
            lower_wick <= body_size):
            shooting_star[i] = True
    
    return hammer, shooting_star

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data ONCE before loop for trend filter
    df_weekly = get_htf_data(prices, '1w')
    
    # Calculate weekly EMA for trend filter
    close_w = df_weekly['close'].values
    ema_w = calculate_ema(close_w, 21)
    ema_w_aligned = align_htf_to_ltf(prices, df_weekly, ema_w)
    
    # Daily data for pattern detection
    open_prices = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate volume moving average
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    # Calculate candlestick patterns
    bullish_engulf, bearish_engulf = calculate_engulfing_signal(open_prices, high, low, close)
    hammer, shooting_star = calculate_hammer_shooting_star(open_prices, high, low, close)
    
    # Combine bullish/bearish signals
    bullish_pattern = bullish_engulf | hammer
    bearish_pattern = bearish_engulf | shooting_star
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(VOLUME_MA_PERIOD, 21) + 1
    
    for i in range(start, n):
        # Skip if weekly trend not available
        if np.isnan(ema_w_aligned[i]):
            if position != 0:
                signals[i] = position * SIGNAL_SIZE
            else:
                signals[i] = 0.0
            continue
        
        # Determine weekly trend: price above EMA = uptrend, below = downtrend
        weekly_uptrend = close[i] > ema_w_aligned[i]
        weekly_downtrend = close[i] < ema_w_aligned[i]
        
        # Volume confirmation
        volume_ok = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Check for reversal patterns with volume and trend alignment
        bullish_signal = bullish_pattern[i] and volume_ok and weekly_uptrend
        bearish_signal = bearish_pattern[i] and volume_ok and weekly_downtrend
        
        # Generate signals
        if position == 0:
            if bullish_signal:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
            elif bearish_signal:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long on bearish reversal or trend change
            if bearish_pattern[i] or not weekly_uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = SIGNAL_SIZE
        elif position == -1:
            # Exit short on bullish reversal or trend change
            if bullish_pattern[i] or not weekly_downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -SIGNAL_SIZE
    
    return signals