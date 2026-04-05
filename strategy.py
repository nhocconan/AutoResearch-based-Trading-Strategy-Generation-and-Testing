#!/usr/bin/env python3
"""
Experiment #8194: 1-hour momentum with 4h trend filter and volume confirmation.
Hypothesis: In 1h timeframe, momentum entries aligned with 4h trend direction (using SMA50) 
and confirmed by volume spikes capture short-term continuations while avoiding counter-trend 
whipsaws. Using 4h for trend direction reduces noise, and restricting to active 
trading hours (08-20 UTC) further improves signal quality. Targeting 60-150 total trades 
over 4 years to balance opportunity with cost efficiency.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_8194_1h_4h_mom_vol_v1"
timeframe = "1h"
leverage = 1.0

# Parameters
RSI_PERIOD = 14
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30
SMA_TREND_PERIOD = 50
VOLUME_SPIKE_MULTIPLIER = 2.0
VOLUME_AVG_PERIOD = 20
SIGNAL_SIZE = 0.20
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate 4h SMA for trend direction
    close_4h = df_4h['close'].values
    sma_4h = pd.Series(close_4h).rolling(window=SMA_TREND_PERIOD, min_periods=SMA_TREND_PERIOD).mean().values
    # 1 = uptrend (close > SMA), -1 = downtrend (close < SMA)
    trend_4h = np.where(close_4h > sma_4h, 1, -1)
    trend_4h_aligned = align_htf_to_ltf(prices, df_4h, trend_4h)
    
    # Calculate LTF indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # RSI for momentum
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/RSI_PERIOD, adjust=False, min_periods=RSI_PERIOD).mean()
    avg_loss = loss.ewm(alpha=1/RSI_PERIOD, adjust=False, min_periods=RSI_PERIOD).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    # Volume spike detector
    volume_ma = pd.Series(volume).rolling(window=VOLUME_AVG_PERIOD, min_periods=VOLUME_AVG_PERIOD).mean().values
    volume_spike = volume > (volume_ma * VOLUME_SPIKE_MULTIPLIER)
    
    # ATR for risk management
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=ATR_PERIOD, adjust=False, min_periods=ATR_PERIOD).mean().values
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(RSI_PERIOD, SMA_TREND_PERIOD, VOLUME_AVG_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if outside trading session
        if not in_session[i]:
            if position != 0:
                # Check stoploss
                if position == 1 and close[i] <= stop_price:
                    signals[i] = 0.0
                    position = 0
                elif position == -1 and close[i] >= stop_price:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = position * SIGNAL_SIZE
            else:
                signals[i] = 0.0
            continue
            
        # Skip if HTF trend data not available
        if np.isnan(trend_4h_aligned[i]):
            if position != 0:
                # Check stoploss
                if position == 1 and close[i] <= stop_price:
                    signals[i] = 0.0
                    position = 0
                elif position == -1 and close[i] >= stop_price:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = position * SIGNAL_SIZE
            else:
                signals[i] = 0.0
            continue
        
        # Check stoploss
        if position == 1:  # long position
            if close[i] <= stop_price:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = SIGNAL_SIZE
                continue
        elif position == -1:  # short position
            if close[i] >= stop_price:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -SIGNAL_SIZE
                continue
        
        # Momentum conditions
        rsi_oversold = rsi[i] < RSI_OVERSOLD
        rsi_overbought = rsi[i] > RSI_OVERBOUGHT
        vol_spike = volume_spike[i] if not np.isnan(volume_ma[i]) else False
        
        # Trend alignment
        bull_trend = trend_4h_aligned[i] == 1
        bear_trend = trend_4h_aligned[i] == -1
        
        # Entry conditions
        long_entry = rsi_oversold and vol_spike and bull_trend
        short_entry = rsi_overbought and vol_spike and bear_trend
        
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
        # Position holding handled above in stoploss check
    
    return signals