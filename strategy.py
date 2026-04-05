# 11:59 AM Sat Apr 5, 2025
# Experiment #8054: 1-hour Price Action with 4h/1d Trend Filter and Volume Confirmation
# Hypothesis: In both bull and bear markets, price tends to continue in the direction of higher timeframe trends.
# Using 1d trend (price above/below 200 EMA) for direction, 4h for momentum confirmation (RSI > 50/< 50),
# and 1h for precise entry on pullbacks to the 20 EMA with volume confirmation.
# This approach should capture trending moves while avoiding counter-trend trades,
# reducing whipsaw in choppy markets. Target: 15-37 trades/year (60-150 total over 4 years).
# Volume confirmation ensures breaks have conviction. Fixed position size of 0.20 limits risk.

#!/usr/bin/env python3

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_8054_1h_4h_1d_trend_ema_volume_v1"
timeframe = "1h"
leverage = 1.0

# Parameters
EMA_200_PERIOD = 200   # 1d trend filter
EMA_20_PERIOD = 20     # 1h entry EMA
RSI_PERIOD = 14        # 4h momentum
VOLUME_MA_PERIOD = 20  # Volume confirmation
VOLUME_THRESHOLD = 1.5 # Volume must be 1.5x average
RSI_LONG_THRESHOLD = 50 # 4h RSI > 50 for bullish momentum
RSI_SHORT_THRESHOLD = 50 # 4h RSI < 50 for bearish momentum
SIGNAL_SIZE = 0.20     # Fixed position size
ATR_PERIOD = 14        # ATR for stop loss
ATR_STOP_MULTIPLIER = 2.5 # Stop loss distance

def generate_signals(prices):
    n = len(prices)
    if n < 250:  # Need enough data for 200 EMA
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate 1d EMA200 for trend direction
    close_1d = df_1d['close'].values
    ema_200 = pd.Series(close_1d).ewm(span=EMA_200_PERIOD, adjust=False, min_periods=EMA_200_PERIOD).mean().values
    
    # Price relative to 200 EMA: above = bullish bias, below = bearish bias
    price_vs_ema200 = np.where(close_1d > ema_200, 1, -1)  # 1=bullish, -1=bearish
    price_vs_ema200_aligned = align_htf_to_ltf(prices, df_1d, price_vs_ema200)
    
    # Calculate 4h RSI for momentum confirmation
    close_4h = df_4h['close'].values
    delta = pd.Series(close_4h).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/RSI_PERIOD, adjust=False, min_periods=RSI_PERIOD).mean()
    avg_loss = loss.ewm(alpha=1/RSI_PERIOD, adjust=False, min_periods=RSI_PERIOD).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    rsi_aligned = align_htf_to_ltf(prices, df_4h, rsi_values)
    
    # Calculate LTF indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1h EMA20 for entry
    ema_20 = pd.Series(close).ewm(span=EMA_20_PERIOD, adjust=False, min_periods=EMA_20_PERIOD).mean().values
    
    # Volume moving average for confirmation
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    # ATR for risk management
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=ATR_PERIOD, adjust=False, min_periods=ATR_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(EMA_200_PERIOD, EMA_20_PERIOD, RSI_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    # Pre-calculate session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    for i in range(start, n):
        # Session filter: only trade between 08:00-20:00 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            # Outside session: flatten position
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
            
        # Skip if HTF data not available
        if np.isnan(price_vs_ema200_aligned[i]) or np.isnan(rsi_aligned[i]):
            # Hold current position or stay flat
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
            continue
            
        # Check stoploss
        if position == 1:  # long position
            if close[i] <= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:  # short position
            if close[i] >= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        # Determine market bias from 1d EMA200
        bull_bias = price_vs_ema200_aligned[i] == 1   # 1d close above EMA200
        bear_bias = price_vs_ema200_aligned[i] == -1  # 1d close below EMA200
        
        # 4h momentum confirmation
        rsi_value = rsi_aligned[i]
        bullish_momentum = rsi_value > RSI_LONG_THRESHOLD
        bearish_momentum = rsi_value < RSI_SHORT_THRESHOLD
        
        # Volume confirmation
        volume_confirmed = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # 1h EMA20 pullback entry
        price_above_ema20 = close[i] > ema_20[i]
        price_below_ema20 = close[i] < ema_20[i]
        
        # Entry conditions
        long_entry = bull_bias and bullish_momentum and volume_confirmed and price_above_ema20
        short_entry = bear_bias and bearish_momentum and volume_confirmed and price_below_ema20
        
        # Generate signals
        if position == 0:
            if long_entry:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif short_entry:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            signals[i] = SIGNAL_SIZE
        elif position == -1:
            signals[i] = -SIGNAL_SIZE
    
    return signals