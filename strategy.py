#1/3
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_Donchian20_1dVolumeSpike_ADX"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for volume spike and ADX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 20-period ATR for Donchian width
    tr1 = np.abs(high[1:] - low[1:])
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 12-period ADX on 1d
    plus_dm = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    minus_dm = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    tr_1d = np.maximum(np.abs(high_1d[1:] - low_1d[1:]), 
                       np.maximum(np.abs(high_1d[1:] - close_1d[:-1]), 
                                  np.abs(low_1d[1:] - close_1d[:-1])))
    tr_1d = np.concatenate([[np.nan], tr_1d])
    
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).sum().values / atr_1d
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).sum().values / atr_1d
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align 1d indicators to 12h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    volume_1d_ma = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_1d_ma_aligned = align_htf_to_ltf(prices, df_1d, volume_1d_ma)
    
    # Calculate Donchian channels (20-period)
    upper_channel = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lower_channel = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 20)  # Need 20 periods for Donchian and volume MA
    
    for i in range(start_idx, n):
        # Skip if required data unavailable (NaN from indicators)
        if np.isnan(adx_aligned[i]) or np.isnan(volume_1d_ma_aligned[i]) or \
           np.isnan(upper_channel[i]) or np.isnan(lower_channel[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        adx_val = adx_aligned[i]
        vol_ma = volume_1d_ma_aligned[i]
        vol_ratio = volume[i] / vol_ma if vol_ma > 0 else 0
        upper = upper_channel[i]
        lower = lower_channel[i]
        
        if position == 0:
            # Enter long: price breaks above upper Donchian + ADX > 25 + volume spike
            if price > upper and adx_val > 25 and vol_ratio > 2.0:
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below lower Donchian + ADX > 25 + volume spike
            elif price < lower and adx_val > 25 and vol_ratio > 2.0:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below lower Donchian OR ADX < 20
            if price < lower or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above upper Donchian OR ADX < 20
            if price > upper or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

#2/3
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_KAMA_1dRSI_VolumeFilter"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for RSI and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 14-period RSI on 1d close
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    avg_gain[0] = gain[0]
    avg_loss[0] = loss[0]
    for i in range(1, len(gain)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss > 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    # Calculate 1d volume moving average
    volume_1d_ma = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_1d_ma_aligned = align_htf_to_ltf(prices, df_1d, volume_1d_ma)
    
    # Calculate KAMA on 12h close
    er = np.zeros(n)
    change = np.abs(np.diff(close, periods=10))
    volatility = np.sum(np.abs(np.diff(close)), axis=1)
    volatility = pd.Series(volatility).rolling(window=10, min_periods=1).sum().values
    er = np.where(volatility > 0, change / volatility, 0)
    
    sc = (er * (0.6667 - 0.0645) + 0.0645) ** 2
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Need 20 periods for volume MA
    
    for i in range(start_idx, n):
        # Skip if required data unavailable (NaN from indicators)
        if np.isnan(rsi_aligned[i]) or np.isnan(volume_1d_ma_aligned[i]) or np.isnan(kama[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        kama_val = kama[i]
        rsi_val = rsi_aligned[i]
        vol_ma = volume_1d_ma_aligned[i]
        vol_ratio = volume[i] / vol_ma if vol_ma > 0 else 0
        
        if position == 0:
            # Enter long: price above KAMA AND RSI < 50 (bullish momentum) AND volume spike
            if price > kama_val and rsi_val < 50 and vol_ratio > 1.5:
                signals[i] = 0.25
                position = 1
            # Enter short: price below KAMA AND RSI > 50 (bearish momentum) AND volume spike
            elif price < kama_val and rsi_val > 50 and vol_ratio > 1.5:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price below KAMA OR RSI > 70 (overbought)
            if price < kama_val or rsi_val > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price above KAMA OR RSI < 30 (oversold)
            if price > kama_val or rsi_val < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

#3/3
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_ADX_Trend_1dVolumeSpike"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for volume spike
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    volume_1d_ma = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_1d_ma_aligned = align_htf_to_ltf(prices, df_1d, volume_1d_ma)
    
    # Calculate 14-period ADX on 12h
    plus_dm = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), 
                       np.maximum(high[1:] - high[:-1], 0), 0)
    minus_dm = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), 
                        np.maximum(low[:-1] - low[1:], 0), 0)
    tr = np.maximum(np.abs(high[1:] - low[1:]), 
                    np.maximum(np.abs(high[1:] - close[:-1]), 
                               np.abs(low[1:] - close[:-1])))
    tr = np.concatenate([[np.nan], tr])
    
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).sum().values / atr
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).sum().values / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 28  # Need 14+14 periods for ADX calculation
    
    for i in range(start_idx, n):
        # Skip if required data unavailable (NaN from indicators)
        if np.isnan(adx[i]) or np.isnan(volume_1d_ma_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        adx_val = adx[i]
        vol_ma = volume_1d_ma_aligned[i]
        vol_ratio = volume[i] / vol_ma if vol_ma > 0 else 0
        
        if position == 0:
            # Enter long: ADX > 25 AND +DI > -DI AND volume spike
            if adx_val > 25 and plus_di[i] > minus_di[i] and vol_ratio > 2.0:
                signals[i] = 0.25
                position = 1
            # Enter short: ADX > 25 AND -DI > +DI AND volume spike
            elif adx_val > 25 and minus_di[i] > plus_di[i] and vol_ratio > 2.0:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: ADX < 20 OR -DI > +DI
            if adx_val < 20 or minus_di[i] > plus_di[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: ADX < 20 OR +DI > -DI
            if adx_val < 20 or plus_di[i] > minus_di[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals