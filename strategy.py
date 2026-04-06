#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h KAMA + RSI + Chop Filter
# Long when KAMA direction is up (KAMA > KAMA prev) AND RSI > 50 AND Chop > 61.8 (ranging market)
# Short when KAMA direction is down (KAMA < KAMA prev) AND RSI < 50 AND Chop > 61.8 (ranging market)
# Exit when Chop < 38.2 (trending market) or opposite signal
# Uses KAMA for adaptive trend, RSI for momentum, Chop for regime filter to avoid whipsaws
# Target: 50-150 total trades over 4 years (12-37/year) for optimal 12h performance

name = "12h_kama_rsi_chop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # KAMA (Adaptive Moving Average)
    # Efficiency Ratio (ER)
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.sum(np.abs(np.diff(close)), axis=0)  # This is incorrect, need to fix
    
    # Let me recalculate properly
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.abs(np.diff(close)).sum()  # Still wrong for rolling
    
    # Better approach: calculate ER properly
    diff = np.diff(close, prepend=close[0])
    change = np.abs(diff)
    # Volatility is sum of absolute changes over period
    volatility = pd.Series(np.abs(diff)).rolling(window=10, min_periods=10).sum().values
    er = np.where(volatility != 0, change / volatility, 0)
    # Smoothing constants
    sc = (er * (0.6667 - 0.0645) + 0.0645) ** 2
    # KAMA calculation
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # RSI (14-period)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean()
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean()
    rs = np.where(avg_loss.values != 0, avg_gain.values / avg_loss.values, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Chop Index (14-period)
    true_range = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    true_range[0] = high[0] - low[0]
    atr = pd.Series(true_range).rolling(window=14, min_periods=14).mean().values
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr.sum() / (highest_high - lowest_low)) / np.log10(14)
    # Fix chop calculation
    atr_sum = pd.Series(true_range).rolling(window=14, min_periods=14).sum().values
    hh_ll_diff = highest_high - lowest_low
    chop = 100 * np.log10(atr_sum / hh_ll_diff) / np.log10(14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        # Skip if required data not available
        if np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits: chop < 38.2 (trending market) or opposite signal
        if chop[i] < 38.2:
            signals[i] = 0.0
            position = 0
        elif position == 1:  # long position
            if kama[i] <= kama[i-1] or rsi[i] < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            if kama[i] >= kama[i-1] or rsi[i] > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries
            # Long: KAMA up AND RSI > 50 AND Chop > 61.8 (ranging)
            if kama[i] > kama[i-1] and rsi[i] > 50 and chop[i] > 61.8:
                signals[i] = 0.25
                position = 1
            # Short: KAMA down AND RSI < 50 AND Chop > 61.8 (ranging)
            elif kama[i] < kama[i-1] and rsi[i] < 50 and chop[i] > 61.8:
                signals[i] = -0.25
                position = -1
    
    return signals

</think>

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian Breakout with Volume and ADX Filter
# Long when price breaks above Donchian(20) upper AND ADX > 25 (trending) AND volume > 1.5x 20-period average
# Short when price breaks below Donchian(20) lower AND ADX > 25 (trending) AND volume > 1.5x 20-period average
# Exit when price crosses Donchian midline OR ADX < 20 (losing momentum)
# Uses Donchian for breakouts, ADX for trend strength, volume for confirmation
# Target: 50-150 total trades over 4 years (12-37/year) for optimal 12h performance

name = "12h_donchian20_adx_vol_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian Channel (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_upper = highest_high
    donchian_lower = lowest_low
    donchian_mid = (donchian_upper + donchian_lower) / 2
    
    # ADX (14-period) for trend strength
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = high[0] - low[0]
    tr2[0] = high[0] - close[0]
    tr3[0] = low[0] - close[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    dm_plus = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                       np.maximum(high - np.roll(high, 1), 0), 0)
    dm_minus = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                        np.maximum(np.roll(low, 1) - low, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values
    tr14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    dm_plus14 = pd.Series(dm_plus).rolling(window=14, min_periods=14).sum().values
    dm_minus14 = pd.Series(dm_minus).rolling(window=14, min_periods=14).sum().values
    
    # DI values
    di_plus = 100 * dm_plus14 / tr14
    di_minus = 100 * dm_minus14 / tr14
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_threshold = 1.5 * volume_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(40, n):
        # Skip if required data not available
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or np.isnan(adx[i]) or np.isnan(volume_threshold[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits
        if position == 1:  # long position
            if close[i] < donchian_mid[i] or adx[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            if close[i] > donchian_mid[i] or adx[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with ADX and volume confirmation
            # Long: price breaks above Donchian upper AND ADX > 25 AND volume confirmation
            if (close[i] > donchian_upper[i] and close[i-1] <= donchian_upper[i-1] and 
                adx[i] > 25 and volume[i] > volume_threshold[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian lower AND ADX > 25 AND volume confirmation
            elif (close[i] < donchian_lower[i] and close[i-1] >= donchian_lower[i-1] and 
                  adx[i] > 25 and volume[i] > volume_threshold[i]):
                signals[i] = -0.25
                position = -1
    
    return signals