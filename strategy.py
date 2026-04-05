#!/usr/bin/env python3
"""
Experiment #8614: 1h RSI divergence + 4h trend filter + volume spike + session filter.
Hypothesis: RSI divergence on 1h captures exhaustion in trending markets, while 4h EMA200 filter ensures alignment with higher timeframe trend. Volume spike confirms institutional interest. Session filter (08-20 UTC) reduces noise. Targets 60-150 total trades over 4 years (15-37/year) to balance opportunity and fee drag.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_8614_1h_rsi_div_4h_trend_vol_sess_v1"
timeframe = "1h"
leverage = 1.0

# Parameters
RSI_PERIOD = 14
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30
DIVERGENCE_LOOKBACK = 10
EMA_TREND_PERIOD = 200
VOLUME_SPIKE_MULTIPLIER = 2.0
VOLUME_MA_PERIOD = 20
SIGNAL_SIZE = 0.20
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5
SESSION_START_HOUR = 8
SESSION_END_HOUR = 20

def calculate_rsi(close, period):
    """Calculate RSI using Wilder's smoothing"""
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate 4h EMA200 for trend filter
    close_4h = df_4h['close'].values
    ema_4h = pd.Series(close_4h).ewm(span=EMA_TREND_PERIOD, adjust=False, min_periods=EMA_TREND_PERIOD).mean().values
    
    # Price relative to 4h EMA200: above = bullish bias, below = bearish bias
    price_vs_ema = np.where(close_4h > ema_4h, 1, 
                     np.where(close_4h < ema_4h, -1, 0))  # 1=bullish, -1=bearish, 0=at EMA
    price_vs_ema_aligned = align_htf_to_ltf(prices, df_4h, price_vs_ema)
    
    # Calculate LTF indicators (1h)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # RSI
    rsi = calculate_rsi(close, RSI_PERIOD)
    
    # Volume moving average
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    # ATR for risk management
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    # Precompute session hours
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(RSI_PERIOD * 2, EMA_TREND_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD) + DIVERGENCE_LOOKBACK
    
    for i in range(start, n):
        # Skip if HTF data not available
        if np.isnan(price_vs_ema_aligned[i]):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
            continue
            
        # Session filter: only trade 08-20 UTC
        hour = hours[i]
        if hour < SESSION_START_HOUR or hour > SESSION_END_HOUR:
            if position != 0:
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
        elif position == -1:  # short position
            if close[i] >= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        # RSI divergence detection
        # Bullish divergence: price makes lower low, RSI makes higher low
        # Bearish divergence: price makes higher high, RSI makes lower high
        lookback_start = max(0, i - DIVERGENCE_LOOKBACK)
        
        # Find recent lows and highs in price
        price_low_idx = np.argmin(low[lookback_start:i+1]) + lookback_start
        price_high_idx = np.argmax(high[lookback_start:i+1]) + lookback_start
        rsi_low_idx = np.argmin(rsi[lookback_start:i+1]) + lookback_start
        rsi_high_idx = np.argmax(rsi[lookback_start:i+1]) + lookback_start
        
        bullish_divergence = (low[price_low_idx] < low[i] and 
                             rsi[rsi_low_idx] > rsi[i] and
                             price_low_idx != i and rsi_low_idx != i)
        
        bearish_divergence = (high[price_high_idx] > high[i] and 
                             rsi[rsi_high_idx] < rsi[i] and
                             price_high_idx != i and rsi_high_idx != i)
        
        # Volume confirmation
        volume_spike = volume[i] > (volume_ma[i] * VOLUME_SPIKE_MULTIPLIER) if not np.isnan(volume_ma[i]) else False
        
        # Determine market bias from 4h EMA200
        bull_bias = price_vs_ema_aligned[i] == 1   # 4h price above EMA200
        bear_bias = price_vs_ema_aligned[i] == -1  # 4h price below EMA200
        
        # Entry conditions
        long_entry = bull_bias and bullish_divergence and volume_spike
        short_entry = bear_bias and bearish_divergence and volume_spike
        
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

</think>
#!/usr/bin/env python3
"""
Experiment #8614: 1h RSI divergence + 4h trend filter + volume spike + session filter.
Hypothesis: RSI divergence on 1h captures exhaustion in trending markets, while 4h EMA200 filter ensures alignment with higher timeframe trend. Volume spike confirms institutional interest. Session filter (08-20 UTC) reduces noise. Targets 60-150 total trades over 4 years (15-37/year) to balance opportunity and fee drag.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_8614_1h_rsi_div_4h_trend_vol_sess_v1"
timeframe = "1h"
leverage = 1.0

# Parameters
RSI_PERIOD = 14
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30
DIVERGENCE_LOOKBACK = 10
EMA_TREND_PERIOD = 200
VOLUME_SPIKE_MULTIPLIER = 2.0
VOLUME_MA_PERIOD = 20
SIGNAL_SIZE = 0.20
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5
SESSION_START_HOUR = 8
SESSION_END_HOUR = 20

def calculate_rsi(close, period):
    """Calculate RSI using Wilder's smoothing"""
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate 4h EMA200 for trend filter
    close_4h = df_4h['close'].values
    ema_4h = pd.Series(close_4h).ewm(span=EMA_TREND_PERIOD, adjust=False, min_periods=EMA_TREND_PERIOD).mean().values
    
    # Price relative to 4h EMA200: above = bullish bias, below = bearish bias
    price_vs_ema = np.where(close_4h > ema_4h, 1, 
                     np.where(close_4h < ema_4h, -1, 0))  # 1=bullish, -1=bearish, 0=at EMA
    price_vs_ema_aligned = align_htf_to_ltf(prices, df_4h, price_vs_ema)
    
    # Calculate LTF indicators (1h)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # RSI
    rsi = calculate_rsi(close, RSI_PERIOD)
    
    # Volume moving average
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    # ATR for risk management
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    # Precompute session hours
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(RSI_PERIOD * 2, EMA_TREND_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD) + DIVERGENCE_LOOKBACK
    
    for i in range(start, n):
        # Skip if HTF data not available
        if np.isnan(price_vs_ema_aligned[i]):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
            continue
            
        # Session filter: only trade 08-20 UTC
        hour = hours[i]
        if hour < SESSION_START_HOUR or hour > SESSION_END_HOUR:
            if position != 0:
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
        elif position == -1:  # short position
            if close[i] >= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        # RSI divergence detection
        # Bullish divergence: price makes lower low, RSI makes higher low
        # Bearish divergence: price makes higher high, RSI makes lower high
        lookback_start = max(0, i - DIVERGENCE_LOOKBACK)
        
        # Find recent lows and highs in price
        price_low_idx = np.argmin(low[lookback_start:i+1]) + lookback_start
        price_high_idx = np.argmax(high[lookback_start:i+1]) + lookback_start
        rsi_low_idx = np.argmin(rsi[lookback_start:i+1]) + lookback_start
        rsi_high_idx = np.argmax(rsi[lookback_start:i+1]) + lookback_start
        
        bullish_divergence = (low[price_low_idx] < low[i] and 
                             rsi[rsi_low_idx] > rsi[i] and
                             price_low_idx != i and rsi_low_idx != i)
        
        bearish_divergence = (high[price_high_idx] > high[i] and 
                             rsi[rsi_high_idx] < rsi[i] and
                             price_high_idx != i and rsi_high_idx != i)
        
        # Volume confirmation
        volume_spike = volume[i] > (volume_ma[i] * VOLUME_SPIKE_MULTIPLIER) if not np.isnan(volume_ma[i]) else False
        
        # Determine market bias from 4h EMA200
        bull_bias = price_vs_ema_aligned[i] == 1   # 4h price above EMA200
        bear_bias = price_vs_ema_aligned[i] == -1  # 4h price below EMA200
        
        # Entry conditions
        long_entry = bull_bias and bullish_divergence and volume_spike
        short_entry = bear_bias and bearish_divergence and volume_spike
        
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