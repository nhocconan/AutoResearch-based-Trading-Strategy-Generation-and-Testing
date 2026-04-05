# 1h RSI mean reversion with 1d trend filter and volume confirmation
# Hypothesis: In BTC/ETH/SOL, RSI extremes during 1d trend regime capture mean reversion moves. 
# 1d EMA filter ensures we only take reversals in direction of higher timeframe trend.
# Volume confirmation filters low-conviction moves. Session filter (08-20 UTC) reduces noise.
# Target: 60-150 trades over 4 years (15-37/year) with disciplined risk management.

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_7794_1h_rsi_meanrev_1d_ema_vol_sess_v1"
timeframe = "1h"
leverage = 1.0

# Parameters
RSI_PERIOD = 14
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30
EMA_TREND = 50
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.20
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA for trend filter
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=EMA_TREND, adjust=False, min_periods=EMA_TREND).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate LTF indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # RSI calculation
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/RSI_PERIOD, adjust=False, min_periods=RSI_PERIOD).mean()
    avg_loss = loss.ewm(alpha=1/RSI_PERIOD, adjust=False, min_periods=RSI_PERIOD).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values  # neutral RSI when no data
    
    # Volume moving average
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    # ATR for risk management
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=ATR_PERIOD, adjust=False, min_periods=ATR_PERIOD).mean().values
    
    # Pre-compute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(RSI_PERIOD, EMA_TREND, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        hour = hours[i]
        in_session = (8 <= hour <= 20)  # UTC 8-20
        
        # Skip if HTF data not available
        if np.isnan(ema_1d_aligned[i]):
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
        
        # Determine market regime from 1d EMA
        bull_regime = close[i] > ema_1d_aligned[i]   # price above 1d EMA
        bear_regime = close[i] < ema_1d_aligned[i]   # price below 1d EMA
        
        # Volume confirmation
        volume_confirmed = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # RSI extreme conditions
        rsi_overbought = rsi[i] >= RSI_OVERBOUGHT
        rsi_oversold = rsi[i] <= RSI_OVERSOLD
        
        # Entry conditions: RSI extreme + volume + session + trend alignment
        long_entry = rsi_oversold and bull_regime and volume_confirmed and in_session
        short_entry = rsi_overbought and bear_regime and volume_confirmed and in_session
        
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