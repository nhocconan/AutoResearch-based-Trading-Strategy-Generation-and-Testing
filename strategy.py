# 1d_KAMA_RSI_Chop_Filter - 4h strategy with KAMA trend + RSI + Chop regime filter
# KAMA adapts to market noise, RSI identifies overbought/oversold, Chop filter avoids ranging markets
# Works in bull (KAMA up + RSI pullback) and bear (KAMA down + RSI bounce) markets
# Target: 20-40 trades/year to minimize fee drag
#!/usr/bin/env python3
name = "1d_KAMA_RSI_Chop_Filter"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load daily data for KAMA, RSI, and Chop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Daily KAMA (adaptive moving average)
    close_1d = df_1d['close'].values
    # Efficiency Ratio
    change = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    volatility = np.sum(np.abs(np.diff(close_1d)), axis=0)  # Will fix below
    # Recalculate properly
    volatility = []
    for i in range(len(close_1d)):
        if i < 1:
            volatility.append(0)
        else:
            volatility.append(np.sum(np.abs(np.diff(close_1d[max(0, i-9):i+1]))))
    volatility = np.array(volatility)
    er = np.where(volatility > 0, change / volatility, 0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1))**2  # fast=2, slow=30
    kama = np.zeros_like(close_1d)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    # Daily RSI
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False).mean().values
    rs = np.where(avg_loss > 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Daily Chopiness Index
    def true_range(h, l, c_prev):
        return np.maximum(h - l, np.maximum(np.abs(h - c_prev), np.abs(l - c_prev)))
    
    tr = np.array([true_range(high[i], low[i], 
                           close_1d[i-1] if i > 0 else close_1d[0]) 
                   for i in range(len(close_1d))])
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    max_h = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_l = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = np.where((max_h - min_l) > 0, 
                    100 * np.log10(atr14.sum() / (max_h - min_l)) / np.log10(14), 
                    50)
    # Fix chop calculation
    chop = []
    for i in range(len(close_1d)):
        if i < 13:
            chop.append(50)
        else:
            tr_sum = np.sum(tr[i-13:i+1])
            range_14 = np.max(high[i-13:i+1]) - np.min(low[i-13:i+1])
            if range_14 > 0:
                chop_val = 100 * np.log10(tr_sum / range_14) / np.log10(14)
            else:
                chop_val = 50
            chop.append(chop_val)
    chop = np.array(chop)
    
    # Align to 4h
    kama_4h = align_htf_to_ltf(prices, df_1d, kama)
    rsi_4h = align_htf_to_ltf(prices, df_1d, rsi)
    chop_4h = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 30
    
    for i in range(start_idx, n):
        if (np.isnan(kama_4h[i]) or np.isnan(rsi_4h[i]) or np.isnan(chop_4h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Chop regime: >61.8 = ranging (mean revert), <38.2 = trending (trend follow)
        # We'll use Chop < 50 as trending regime for simplicity
        trending_regime = chop_4h[i] < 50
        
        if position == 0:
            if trending_regime:
                # In trending regime: follow KAMA direction with RSI pullback
                # Long: price above KAMA and RSI < 40 (pullback in uptrend)
                if close[i] > kama_4h[i] and rsi_4h[i] < 40:
                    signals[i] = 0.25
                    position = 1
                # Short: price below KAMA and RSI > 60 (bounce in downtrend)
                elif close[i] < kama_4h[i] and rsi_4h[i] > 60:
                    signals[i] = -0.25
                    position = -1
            else:
                # In ranging regime: mean reversion at extremes
                # Long: RSI < 30 (oversold)
                if rsi_4h[i] < 30:
                    signals[i] = 0.20
                    position = 1
                # Short: RSI > 70 (overbought)
                elif rsi_4h[i] > 70:
                    signals[i] = -0.20
                    position = -1
        elif position == 1:
            # Exit conditions
            if trending_regime:
                # Exit trend: price crosses below KAMA or RSI > 70
                if close[i] < kama_4h[i] or rsi_4h[i] > 70:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:
                # Exit range: RSI returns to neutral
                if 40 <= rsi_4h[i] <= 60:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.20
        elif position == -1:
            # Exit conditions
            if trending_regime:
                # Exit trend: price crosses above KAMA or RSI < 30
                if close[i] > kama_4h[i] or rsi_4h[i] < 30:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
            else:
                # Exit range: RSI returns to neutral
                if 40 <= rsi_4h[i] <= 60:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.20
    
    return signals

#!/usr/bin/env python3
name = "1d_KAMA_RSI_Chop_Filter"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load daily data for KAMA, RSI, and Chop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Daily KAMA (adaptive moving average)
    close_1d = df_1d['close'].values
    # Efficiency Ratio
    change = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    volatility = []
    for i in range(len(close_1d)):
        if i < 1:
            volatility.append(0)
        else:
            volatility.append(np.sum(np.abs(np.diff(close_1d[max(0, i-9):i+1]))))
    volatility = np.array(volatility)
    er = np.where(volatility > 0, change / volatility, 0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1))**2  # fast=2, slow=30
    kama = np.zeros_like(close_1d)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    # Daily RSI
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False).mean().values
    rs = np.where(avg_loss > 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Daily Chopiness Index
    def true_range(h, l, c_prev):
        return np.maximum(h - l, np.maximum(np.abs(h - c_prev), np.abs(l - c_prev)))
    
    tr = np.array([true_range(high[i], low[i], 
                           close_1d[i-1] if i > 0 else close_1d[0]) 
                   for i in range(len(close_1d))])
    chop = []
    for i in range(len(close_1d)):
        if i < 13:
            chop.append(50)
        else:
            tr_sum = np.sum(tr[i-13:i+1])
            range_14 = np.max(high[i-13:i+1]) - np.min(low[i-13:i+1])
            if range_14 > 0:
                chop_val = 100 * np.log10(tr_sum / range_14) / np.log10(14)
            else:
                chop_val = 50
            chop.append(chop_val)
    chop = np.array(chop)
    
    # Align to 4h
    kama_4h = align_htf_to_ltf(prices, df_1d, kama)
    rsi_4h = align_htf_to_ltf(prices, df_1d, rsi)
    chop_4h = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 30
    
    for i in range(start_idx, n):
        if (np.isnan(kama_4h[i]) or np.isnan(rsi_4h[i]) or np.isnan(chop_4h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Chop regime: >61.8 = ranging (mean revert), <38.2 = trending (trend follow)
        # We'll use Chop < 50 as trending regime for simplicity
        trending_regime = chop_4h[i] < 50
        
        if position == 0:
            if trending_regime:
                # In trending regime: follow KAMA direction with RSI pullback
                # Long: price above KAMA and RSI < 40 (pullback in uptrend)
                if close[i] > kama_4h[i] and rsi_4h[i] < 40:
                    signals[i] = 0.25
                    position = 1
                # Short: price below KAMA and RSI > 60 (bounce in downtrend)
                elif close[i] < kama_4h[i] and rsi_4h[i] > 60:
                    signals[i] = -0.25
                    position = -1
            else:
                # In ranging regime: mean reversion at extremes
                # Long: RSI < 30 (oversold)
                if rsi_4h[i] < 30:
                    signals[i] = 0.20
                    position = 1
                # Short: RSI > 70 (overbought)
                elif rsi_4h[i] > 70:
                    signals[i] = -0.20
                    position = -1
        elif position == 1:
            # Exit conditions
            if trending_regime:
                # Exit trend: price crosses below KAMA or RSI > 70
                if close[i] < kama_4h[i] or rsi_4h[i] > 70:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:
                # Exit range: RSI returns to neutral
                if 40 <= rsi_4h[i] <= 60:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.20
        elif position == -1:
            # Exit conditions
            if trending_regime:
                # Exit trend: price crosses above KAMA or RSI < 30
                if close[i] > kama_4h[i] or rsi_4h[i] < 30:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
            else:
                # Exit range: RSI returns to neutral
                if 40 <= rsi_4h[i] <= 60:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.20
    
    return signals

#!/usr/bin/env python3
name = "1d_KAMA_RSI_Chop_Filter"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load daily data for KAMA, RSI, and Chop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Daily KAMA (adaptive moving average)
    close_1d = df_1d['close'].values
    # Efficiency Ratio
    change = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    volatility = []
    for i in range(len(close_1d)):
        if i < 1:
            volatility.append(0)
        else:
            volatility.append(np.sum(np.abs(np.diff(close_1d[max(0, i-9):i+1]))))
    volatility = np.array(volatility)
    er = np.where(volatility > 0, change / volatility, 0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1))**2  # fast=2, slow=30
    kama = np.zeros_like(close_1d)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    # Daily RSI
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False).mean().values
    rs = np.where(avg_loss > 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Daily Chopiness Index
    def true_range(h, l, c_prev):
        return np.maximum(h - l, np.maximum(np.abs(h - c_prev), np.abs(l - c_prev)))
    
    tr = np.array([true_range(high[i], low[i], 
                           close_1d[i-1] if i > 0 else close_1d[0]) 
                   for i in range(len(close_1d))])
    chop = []
    for i in range(len(close_1d)):
        if i < 13:
            chop.append(50)
        else:
            tr_sum = np.sum(tr[i-13:i+1])
            range_14 = np.max(high[i-13:i+1]) - np.min(low[i-13:i+1])
            if range_14 > 0:
                chop_val = 100 * np.log10(tr_sum / range_14) / np.log10(14)
            else:
                chop_val = 50
            chop.append(chop_val)
    chop = np.array(chop)
    
    # Align to 4h
    kama_4h = align_htf_to_ltf(prices, df_1d, kama)
    rsi_4h = align_htf_to_ltf(prices, df_1d, rsi)
    chop_4h = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 30
    
    for i in range(start_idx, n):
        if (np.isnan(kama_4h[i]) or np.isnan(rsi_4h[i]) or np.isnan(chop_4h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Chop regime: >61.8 = ranging (mean revert), <38.2 = trending (trend follow)
        # We'll use Chop < 50 as trending regime for simplicity
        trending_regime = chop_4h[i] < 50
        
        if position == 0:
            if trending_regime:
                # In trending regime: follow KAMA direction with RSI pullback
                # Long: price above KAMA and RSI < 40 (pullback in uptrend)
                if close[i] > kama_4h[i] and rsi_4h[i] < 40:
                    signals[i] = 0.25
                    position = 1
                # Short: price below KAMA and RSI > 60 (bounce in downtrend)
                elif close[i] < kama_4h[i] and rsi_4h[i] > 60:
                    signals[i] = -0.25
                    position = -1
            else:
                # In ranging regime: mean reversion at extremes
                # Long: RSI < 30 (oversold)
                if rsi_4h[i] < 30:
                    signals[i] = 0.20
                    position = 1
                # Short: RSI > 70 (overbought)
                elif rsi_4h[i] > 70:
                    signals[i] = -0.20
                    position = -1
        elif position == 1:
            # Exit conditions
            if trending_regime:
                # Exit trend: price crosses below KAMA or RSI > 70
                if close[i] < kama_4h[i] or rsi_4h[i] > 70:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:
                # Exit range: RSI returns to neutral
                if 40 <= rsi_4h[i] <= 60:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.20
        elif position == -1:
            # Exit conditions
            if trending_regime:
                # Exit trend: price crosses above KAMA or RSI < 30
                if close[i] > kama_4h[i] or rsi_4h[i] < 30:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
            else:
                # Exit range: RSI returns to neutral
                if 40 <= rsi_4h[i] <= 60:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.20
    
    return signals

#!/usr/bin/env python3
name = "1d_KAMA_RSI_Chop_Filter"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load daily data for KAMA, RSI, and Chop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Daily KAMA (adaptive moving average)
    close_1d = df_1d['close'].values
    # Efficiency Ratio
    change = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    volatility = []
    for i in range(len(close_1d)):
        if i < 1:
            volatility.append(0)
        else:
            volatility.append(np.sum(np.abs(np.diff(close_1d[max(0, i-9):i+1]))))
    volatility = np.array(volatility)
    er = np.where(volatility > 0, change / volatility, 0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1))**2  # fast=2, slow=30
    kama = np.zeros_like(close_1d)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    # Daily RSI
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False).mean().values
    rs = np.where(avg_loss > 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Daily Chopiness Index
    def true_range(h, l, c_prev):
        return np.maximum(h - l, np.maximum(np.abs(h - c_prev), np.abs(l - c_prev)))
    
    tr = np.array([true_range(high[i], low[i], 
                           close_1d[i-1] if i > 0 else close_1d[0]) 
                   for i in range(len(close_1d))])
    chop = []
    for i in range(len(close_1d)):
        if i < 13:
            chop.append(50)
        else:
            tr_sum = np.sum(tr[i-13:i+1])
            range_14 = np.max(high[i-13:i+1]) - np.min(low[i-13:i+1])
            if range_14 > 0:
                chop_val = 100 * np.log10(tr_sum / range_14) / np.log10(14)
            else:
                chop_val = 50
            chop.append(chop_val)
    chop = np.array(chop)
    
    # Align to 4h
    kama_4h = align_htf_to_ltf(prices, df_1d, kama)
    rsi_4h = align_htf_to_ltf(prices, df_1d, rsi)
    chop_4h = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 30
    
    for i in range(start_idx, n):
        if (np.isnan(kama_4h[i]) or np.isnan(rsi_4h[i]) or np.isnan(chop_4h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Chop regime: >61.8 = ranging (mean revert), <38.2 = trending (trend follow)
        # We'll use Chop < 50 as trending regime for simplicity
        trending_regime = chop_4h[i] < 50
        
        if position == 0:
            if trending_regime:
                # In trending regime: follow KAMA direction with RSI pullback
                # Long: price above KAMA and RSI < 40 (pullback in uptrend)
                if close[i] > kama_4h[i] and rsi_4h[i] < 40:
                    signals[i] = 0.25
                    position = 1
                # Short: price below KAMA and RSI > 60 (bounce in downtrend)
                elif close[i] < kama_4h[i] and rsi_4h[i] > 60:
                    signals[i] = -0.25
                    position = -1
            else:
                # In ranging regime: mean reversion at extremes
                # Long: RSI < 30 (oversold)
                if rsi_4h[i] < 30:
                    signals[i] = 0.20
                    position = 1
                # Short: RSI > 70 (overbought)
                elif rsi_4h[i] > 70:
                    signals[i] = -0.20
                    position = -1
        elif position == 1:
            # Exit conditions
            if trending_regime:
                # Exit trend: price crosses below KAMA or RSI > 70
                if close[i] < kama_4h[i] or rsi_4h[i] > 70:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:
                # Exit range: RSI returns to neutral
                if 40 <= rsi_4h[i] <= 60:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.20
        elif position == -1:
            # Exit conditions
            if trending_regime:
                # Exit trend: price crosses above KAMA or RSI < 30
                if close[i] > kama_4h[i] or rsi_4h[i] < 30:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
            else:
                # Exit range: RSI returns to neutral
                if 40 <= rsi_4h[i] <= 60:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.20
    
    return signals

#!/usr/bin/env python3
name = "1d_KAMA_RSI_Chop_Filter"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load daily data for KAMA, RSI, and Chop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Daily KAMA (adaptive moving average)
    close_1d = df_1d['close'].values
    # Efficiency Ratio
    change = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    volatility = []
    for i in range(len(close_1d)):
        if i < 1:
            volatility.append(0)
        else:
            volatility.append(np.sum(np.abs(np.diff(close_1d[max(0, i-9):i+1]))))
    volatility = np.array(volatility)
    er = np.where(volatility > 0, change / volatility, 0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1))**2  # fast=2, slow=30
    kama = np.zeros_like(close_1d)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    # Daily RSI
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False).mean().values
    rs = np.where(avg_loss > 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Daily Chopiness Index
    def true_range(h, l, c_prev):
        return np.maximum(h - l, np.maximum(np.abs(h - c_prev), np.abs(l - c_prev)))
    
    tr = np.array([true_range(high[i], low[i], 
                           close_1d[i-1] if i > 0 else close_1d[0]) 
                   for i in range(len(close_1d))])
    chop = []
    for i in range(len(close_1d)):
        if i < 13:
            chop.append(50)
        else:
            tr_sum = np.sum(tr[i-13:i+1])
            range_14 = np.max(high[i-13:i+1]) - np.min(low[i-13:i+1])
            if range_14 > 0:
                chop_val = 100 * np.log10(tr_sum / range_14) / np.log10(14)
            else:
                chop_val = 50
            chop.append(chop_val)
    chop = np.array(chop)
    
    # Align to 4h
    kama_4h = align_htf_to_ltf(prices, df_1d, kama)
    rsi_4h = align_htf_to_ltf(prices, df_1d, rsi)
    chop_4h = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 30
    
    for i in range(start_idx, n):
        if (np.isnan(kama_4h[i]) or np.isnan(rsi_4h[i]) or np.isnan(chop_4h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Chop regime: >61.8 = ranging (mean revert), <38.2 = trending (trend follow)
        # We'll use Chop < 5