#!/usr/bin/env python3
"""
Experiment #7974: 1-hour momentum with 4h trend filter and 1d volume regime.
Hypothesis: In trending markets (4h price above/below 200 EMA), momentum bursts on 1h 
(RSI > 60 for long, RSI < 40 for short) with volume > 1.5x 20-period MA capture 
continuation. Uses 1d volume regime (high/low volatility filter) to avoid chop. 
Restricts to active session (08-20 UTC) to reduce noise. Target: 60-150 trades over 4 years.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_7974_1h_momentum_4h_200ema_1d_vol_regime_v1"
timeframe = "1h"
leverage = 1.0

# Parameters
RSI_PERIOD = 14
RSI_LONG_THRESH = 60
RSI_SHORT_THRESH = 40
EMA_TREND_PERIOD = 200
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
VOLUME_REGIME_PERIOD = 50  # 1d period for volume regime (approx 50 periods for 1h data)
VOLUME_REGIME_THRESHOLD = 1.2  # volume > 1.2x 50-period MA = high vol regime
SIGNAL_SIZE = 0.20
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 4h EMA200 for trend filter
    close_4h = df_4h['close'].values
    ema_4h = pd.Series(close_4h).ewm(span=EMA_TREND_PERIOD, adjust=False, min_periods=EMA_TREND_PERIOD).mean().values
    price_vs_ema_4h = np.where(close_4h > ema_4h, 1, -1)  # 1=uptrend, -1=downtrend
    ema_trend_4h = align_htf_to_ltf(prices, df_4h, price_vs_ema_4h)
    
    # Calculate 1d volume regime (high/low volatility filter)
    volume_1d = df_1d['volume'].values
    volume_ma_1d = pd.Series(volume_1d).rolling(window=VOLUME_REGIME_PERIOD, min_periods=VOLUME_REGIME_PERIOD).mean().values
    volume_ratio_1d = volume_1d / volume_ma_1d  # current volume vs average
    volume_regime = align_htf_to_ltf(prices, df_1d, volume_ratio_1d)
    
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
    
    # Volume confirmation (1h)
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    # ATR for risk management
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=ATR_PERIOD, adjust=False, min_periods=ATR_PERIOD).mean().values
    
    # Session filter: 08-20 UTC (already datetime64 index)
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(RSI_PERIOD, EMA_TREND_PERIOD, VOLUME_MA_PERIOD, VOLUME_REGIME_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if HTF data not available
        if np.isnan(ema_trend_4h[i]) or np.isnan(volume_regime[i]):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
            continue
        
        # Check session (08-20 UTC)
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
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
        
        # Determine market bias from 4h EMA200
        bull_bias = ema_trend_4h[i] == 1   # 4h close above EMA200
        bear_bias = ema_trend_4h[i] == -1  # 4h close below EMA200
        
        # Volume confirmation (1h)
        volume_confirmed = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Volume regime filter (1d): only trade in high volatility regimes
        high_vol_regime = volume_regime[i] > VOLUME_REGIME_THRESHOLD if not np.isnan(volume_regime[i]) else False
        
        # RSI momentum conditions
        rsi_long = rsi[i] > RSI_LONG_THRESH and not np.isnan(rsi[i])
        rsi_short = rsi[i] < RSI_SHORT_THRESH and not np.isnan(rsi[i])
        
        # Entry conditions
        long_entry = bull_bias and rsi_long and volume_confirmed and high_vol_regime and in_session
        short_entry = bear_bias and rsi_short and volume_confirmed and high_vol_regime and in_session
        
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