#!/usr/bin/env python3
"""
4h RSI with Volume Confirmation and Trend Filter
Hypothesis: RSI mean reversion combined with volume spikes and trend direction (via 200 EMA) provides
high-probability entries. Works in bull markets (buy oversold dips) and bear markets (sell overbought
bounces). Volume confirmation ensures institutional participation. Target: 100-200 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "14461_4h_rsi_volume_trend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Load 1d data for EMA200 trend filter (once before loop)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA200 on daily close
    ema200_1d = pd.Series(close_1d).ewm(span=200, min_periods=200).mean().values
    ema200_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # 4h data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(14, n):
        if np.isnan(rsi[i]) or np.isnan(ema200_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits
        if position == 1:  # long
            if rsi[i] >= 70 or close[i] <= entry_price - 2.0 * (np.abs(high - low)).mean():
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short
            if rsi[i] <= 30 or close[i] >= entry_price + 2.0 * (np.abs(high - low)).mean():
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Long setup: RSI < 30 (oversold), price above EMA200 (uptrend), volume spike
            long_setup = (rsi[i] < 30) and (close[i] > ema200_aligned[i]) and vol_filter[i]
            # Short setup: RSI > 70 (overbought), price below EMA200 (downtrend), volume spike
            short_setup = (rsi[i] > 70) and (close[i] < ema200_aligned[i]) and vol_filter[i]
            
            if long_setup:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            elif short_setup:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals

</think>

#!/usr/bin/env python3
"""
4h Donchian Breakout with Volume Confirmation and ATR Filter
Hypothesis: Donchian channel breakouts capture momentum bursts. Volume confirmation ensures
institutional participation. ATR filter avoids choppy markets. Works in bull (breakout longs)
and bear (breakout shorts). Target: 100-200 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "14461_4h_donchian_volume_atr_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for EMA200 trend filter (once before loop)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA200 on daily close
    ema200_1d = pd.Series(close_1d).ewm(span=200, min_periods=200).mean().values
    ema200_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # 4h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channel (20-period)
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max()
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min()
    donchian_high = high_roll.values
    donchian_low = low_roll.values
    
    # ATR(14) for volatility filter and stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(20, n):
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or \
           np.isnan(ema200_aligned[i]) or np.isnan(atr[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits
        if position == 1:  # long
            if close[i] <= donchian_low[i] or close[i] <= entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short
            if close[i] >= donchian_high[i] or close[i] >= entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Long setup: break above Donchian high, price above EMA200 (uptrend), volume spike
            long_setup = (close[i] > donchian_high[i]) and (close[i] > ema200_aligned[i]) and vol_filter[i]
            # Short setup: break below Donchian low, price below EMA200 (downtrend), volume spike
            short_setup = (close[i] < donchian_low[i]) and (close[i] < ema200_aligned[i]) and vol_filter[i]
            
            if long_setup:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            elif short_setup:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals

</think>

#!/usr/bin/env python3
"""
4h RSI Divergence with Volume Confirmation
Hypothesis: RSI divergences signal weakening momentum and impending reversals.
Combined with volume spikes for confirmation and EMA200 trend filter for direction.
Works in bull (bullish divergence longs) and bear (bearish divergence shorts).
Target: 80-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "14461_4h_rsi_divergence_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for EMA200 trend filter (once before loop)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA200 on daily close
    ema200_1d = pd.Series(close_1d).ewm(span=200, min_periods=200).mean().values
    ema200_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # 4h data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Lookback period for divergence detection
    lookback = 10
    
    for i in range(lookback + 14, n):
        if np.isnan(rsi[i]) or np.isnan(ema200_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits
        if position == 1:  # long
            if rsi[i] >= 70 or close[i] <= entry_price - 2.0 * (np.abs(high - low)).mean():
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short
            if rsi[i] <= 30 or close[i] >= entry_price + 2.0 * (np.abs(high - low)).mean():
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Bullish divergence: price makes lower low, RSI makes higher low
            price_lower_low = close[i] < close[i-lookback] and low[i] < low[i-lookback]
            rsi_higher_low = rsi[i] > rsi[i-lookback]
            bullish_div = price_lower_low and rsi_higher_low and vol_filter[i]
            
            # Bearish divergence: price makes higher high, RSI makes lower high
            price_higher_high = close[i] > close[i-lookback] and high[i] > high[i-lookback]
            rsi_lower_high = rsi[i] < rsi[i-lookback]
            bearish_div = price_higher_high and rsi_lower_high and vol_filter[i]
            
            # Trend filter: long only above EMA200, short only below EMA200
            if bullish_div and close[i] > ema200_aligned[i]:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            elif bearish_div and close[i] < ema200_aligned[i]:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals

---</think>#!/usr/bin/env python3
"""
4h MACD Histogram Reversal with Volume Confirmation
Hypothesis: MACD histogram divergences signal momentum exhaustion. Combined with
volume spikes and EMA200 trend filter for high-probability reversals.
Works in bull (buy weakness in uptrend) and bear (sell strength in downtrend).
Target: 80-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "14461_4h_macd_histogram_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for EMA200 trend filter (once before loop)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA200 on daily close
    ema200_1d = pd.Series(close_1d).ewm(span=200, min_periods=200).mean().values
    ema200_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # 4h data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # MACD (12,26,9)
    ema12 = pd.Series(close).ewm(span=12, min_periods=12).mean().values
    ema26 = pd.Series(close).ewm(span=26, min_periods=26).mean().values
    macd_line = ema12 - ema26
    signal_line = pd.Series(macd_line).ewm(span=9, min_periods=9).mean().values
    macd_hist = macd_line - signal_line
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(30, n):
        if np.isnan(ema200_aligned[i]) or np.isnan(macd_hist[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits
        if position == 1:  # long
            if macd_hist[i] < 0 or close[i] <= entry_price - 2.0 * (np.abs(high - low)).mean():
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short
            if macd_hist[i] > 0 or close[i] >= entry_price + 2.0 * (np.abs(high - low)).mean():
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Bullish setup: MACD histogram turning positive from negative, price above EMA200, volume spike
            bullish_setup = (macd_hist[i] > 0 and macd_hist[i-1] <= 0) and (close[i] > ema200_aligned[i]) and vol_filter[i]
            # Bearish setup: MACD histogram turning negative from positive, price below EMA200, volume spike
            bearish_setup = (macd_hist[i] < 0 and macd_hist[i-1] >= 0) and (close[i] < ema200_aligned[i]) and vol_filter[i]
            
            if bullish_setup:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            elif bearish_setup:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals

---