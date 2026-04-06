#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h ADX + RSI with 1d EMA200 trend filter
# Long when ADX > 25 (trending), RSI < 40 (pullback), and price > 1d EMA200 (uptrend)
# Short when ADX > 25, RSI > 60 (pullback), and price < 1d EMA200 (downtrend)
# Uses ADX for trend strength, RSI for pullback entries in trending markets
# EMA200 filter ensures trades align with long-term trend
# Volume confirmation (>1.3x average) to filter false signals
# Target: 60-150 total trades over 4 years with controlled risk
# ATR-based stoploss to limit drawdown (2x ATR)

name = "6h_adx_rsi_1d_ema200_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for EMA200 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # EMA200 calculation
    ema200_1d = pd.Series(close_1d).ewm(span=200, min_periods=200, adjust=False).mean().values
    
    # Align 1d EMA200 to 6h timeframe
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # ADX calculation (14-period)
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    # Directional Movement
    dm_plus = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                       np.maximum(high - np.roll(high, 1), 0), 0)
    dm_minus = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                        np.maximum(np.roll(low, 1) - low, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values
    def smooth(values, period):
        smoothed = np.zeros_like(values)
        smoothed[period-1] = np.mean(values[:period])
        for i in range(period, len(values)):
            smoothed[i] = (smoothed[i-1] * (period-1) + values[i]) / period
        return smoothed
    
    atr = smooth(tr, 14)
    dm_plus_smooth = smooth(dm_plus, 14)
    dm_minus_smooth = smooth(dm_minus, 14)
    
    # DI values
    di_plus = np.where(atr > 0, 100 * dm_plus_smooth / atr, 0)
    di_minus = np.where(atr > 0, 100 * dm_minus_smooth / atr, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) > 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = smooth(dx, 14)
    
    # RSI (14-period)
    delta = np.diff(close)
    delta = np.insert(delta, 0, 0)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    avg_gain[13] = np.mean(gain[1:14])
    avg_loss[13] = np.mean(loss[1:14])
    
    for i in range(14, len(gain)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss > 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume average (20-period)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(14, n):
        # Skip if required data not available
        if (np.isnan(ema200_1d_aligned[i]) or np.isnan(adx[i]) or 
            np.isnan(rsi[i]) or np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 2 * ATR approximation using price range
            if close[i] < entry_price - 2.0 * (high[i] - low[i]):
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: RSI reverses or trend weakens
            elif rsi[i] > 60 or adx[i] < 20:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Stoploss: 2 * ATR approximation
            if close[i] > entry_price + 2.0 * (high[i] - low[i]):
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: RSI reverses or trend weakens
            elif rsi[i] < 40 or adx[i] < 20:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume confirmation
            # Long: ADX > 25 (trending), RSI < 40 (pullback), uptrend, volume spike
            if (adx[i] > 25 and 
                rsi[i] < 40 and
                close[i] > ema200_1d_aligned[i] and
                volume[i] > 1.3 * volume_ma[i]):
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: ADX > 25, RSI > 60 (pullback), downtrend, volume spike
            elif (adx[i] > 25 and 
                  rsi[i] > 60 and
                  close[i] < ema200_1d_aligned[i] and
                  volume[i] > 1.3 * volume_ma[i]):
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R + CCI with 1d EMA50 trend filter
# Long when Williams %R < -80 (oversold), CCI < -100 (strong oversold), and price > 1d EMA50 (uptrend)
# Short when Williams %R > -20 (overbought), CCI > 100 (strong overbought), and price < 1d EMA50 (downtrend)
# Uses momentum oscillators for extreme reversal points in trending markets
# EMA50 filter ensures trades align with intermediate-term trend
# Volume confirmation (>1.4x average) to filter false signals
# Target: 70-180 total trades over 4 years with controlled risk
# ATR-based stoploss to limit drawdown (2x ATR)

name = "6h_williamsr_cci_1d_ema50_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # EMA50 calculation
    ema50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    
    # Align 1d EMA50 to 6h timeframe
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Williams %R (14-period)
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = np.where((highest_high - lowest_low) > 0, 
                          ((highest_high - close) / (highest_high - lowest_low)) * -100, -50)
    
    # CCI (20-period)
    # CCI = (Typical Price - SMA(TP)) / (0.015 * Mean Deviation)
    typical_price = (high + low + close) / 3
    sma_tp = pd.Series(typical_price).rolling(window=20, min_periods=20).mean().values
    
    # Mean deviation
    mean_dev = np.zeros_like(typical_price)
    for i in range(19, len(typical_price)):
        tp_slice = typical_price[i-19:i+1]
        sma_slice = sma_tp[i]
        mean_dev[i] = np.mean(np.abs(tp_slice - sma_slice))
    
    cci = np.where(mean_dev > 0, (typical_price - sma_tp) / (0.015 * mean_dev), 0)
    
    # Volume average (20-period)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(20, n):  # Start after CCI warmup
        # Skip if required data not available
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(cci[i]) or np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 2 * ATR approximation using price range
            if close[i] < entry_price - 2.0 * (high[i] - low[i]):
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: Williams %R reverses or CCI normalizes
            elif williams_r[i] > -50 or abs(cci[i]) < 50:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Stoploss: 2 * ATR approximation
            if close[i] > entry_price + 2.0 * (high[i] - low[i]):
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: Williams %R reverses or CCI normalizes
            elif williams_r[i] < -50 or abs(cci[i]) < 50:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume confirmation
            # Long: Williams %R < -80 (oversold), CCI < -100 (strong oversold), uptrend, volume spike
            if (williams_r[i] < -80 and 
                cci[i] < -100 and
                close[i] > ema50_1d_aligned[i] and
                volume[i] > 1.4 * volume_ma[i]):
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: Williams %R > -20 (overbought), CCI > 100 (strong overbought), downtrend, volume spike
            elif (williams_r[i] > -20 and 
                  cci[i] > 100 and
                  close[i] < ema50_1d_aligned[i] and
                  volume[i] > 1.4 * volume_ma[i]):
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Stochastic RSI with 1d ADX trend filter
# Long when StochRSI < 0.1 (oversold), ADX > 20 (trending), and price > 1d EMA100 (uptrend)
# Short when StochRSI > 0.9 (overbought), ADX > 20 (trending), and price < 1d EMA100 (downtrend)
# Uses StochRSI for extreme momentum readings in trending markets
# ADX filter ensures trades occur only when trend is strong enough
# EMA100 filter ensures alignment with medium-term trend
# Volume confirmation (>1.35x average) to filter false signals
# Target: 65-165 total trades over 4 years with controlled risk
# ATR-based stoploss to limit drawdown (2x ATR)

name = "6h_stochrsi_1d_adx_ema100_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for ADX and EMA100
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 100:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # EMA100 calculation
    ema100_1d = pd.Series(close_1d).ewm(span=100, min_periods=100, adjust=False).mean().values
    
    # Align 1d EMA100 to 6h timeframe
    ema100_1d_aligned = align_htf_to_ltf(prices, df_1d, ema100_1d)
    
    # ADX calculation (14-period) on 1d
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values
    def smooth(values, period):
        smoothed = np.zeros_like(values)
        if len(values) >= period:
            smoothed[period-1] = np.mean(values[:period])
            for i in range(period, len(values)):
                smoothed[i] = (smoothed[i-1] * (period-1) + values[i]) / period
        return smoothed
    
    atr_1d = smooth(tr, 14)
    dm_plus_smooth = smooth(dm_plus, 14)
    dm_minus_smooth = smooth(dm_minus, 14)
    
    # DI values
    di_plus = np.where(atr_1d > 0, 100 * dm_plus_smooth / atr_1d, 0)
    di_minus = np.where(atr_1d > 0, 100 * dm_minus_smooth / atr_1d, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) > 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx_1d = smooth(dx, 14)
    
    # Align 1d ADX to 6h timeframe
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Stochastic RSI (14-period RSI, then Stochastic on RSI)
    # First calculate RSI (14-period)
    delta = np.diff(close)
    delta = np.insert(delta, 0, 0)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    if len(gain) >= 14:
        avg_gain[13] = np.mean(gain[1:14])
        avg_loss[13] = np.mean(loss[1:14])
        
        for i in range(14, len(gain)):
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss > 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Then calculate Stochastic of RSI (14-period)
    rsi_high = pd.Series(rsi).rolling(window=14, min_periods=14).max().values
    rsi_low = pd.Series(rsi).rolling(window=14, min_periods=14).min().values
    stochrsi = np.where((rsi_high - rsi_low) > 0, 
                        (rsi - rsi_low) / (rsi_high - rsi_low), 0.5)
    
    # Volume average (20-period)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(14, n):  # Start after RSI warmup
        # Skip if required data not available
        if (np.isnan(ema100_1d_aligned[i]) or np.isnan(adx_1d_aligned[i]) or 
            np.isnan(stochrsi[i]) or np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 2 * ATR approximation using price range
            if close[i] < entry_price - 2.0 * (high[i] - low[i]):
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: StochRSI reverses or trend weakens
            elif stochrsi[i] > 0.5 or adx_1d_aligned[i] < 15:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Stoploss: 2 * ATR approximation
            if close[i] > entry_price + 2.0 * (high[i] - low[i]):
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: StochRSI reverses or trend weakens
            elif stochrsi[i] < 0.5 or adx_1d_aligned[i] < 15:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume confirmation
            # Long: StochRSI < 0.1 (oversold), ADX > 20 (trending), uptrend, volume spike
            if (stochrsi[i] < 0.1 and 
                adx_1d_aligned[i] > 20 and
                close[i] > ema100_1d_aligned[i] and
                volume[i] > 1.35 * volume_ma[i]):
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: StochRSI > 0.9 (overbought), ADX > 20 (trending), downtrend, volume spike
            elif (stochrsi[i] > 0.9 and 
                  adx_1d_aligned[i] > 20 and
                  close[i] < ema100_1d_aligned[i] and
                  volume[i] > 1.35 * volume_ma[i]):
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian channel breakout with 1d volatility filter and volume confirmation
# Long when price breaks above 20-period Donchian high, 1d ATR ratio > 1.2 (expanding volatility), and volume > 1.5x average
# Short when price breaks below 20-period Donchian low, 1d ATR ratio > 1.2, and volume > 1.5x average
# Uses Donchian breakouts for trend continuation, volatility filter to avoid ranging markets
# Volume confirmation to validate breakout strength
# ATR-based stoploss to limit drawdown (2x ATR)
# Target: 60-160 total trades over 4 years with controlled risk

name = "6h_donchian20_1d_volatility_filter_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for ATR-based volatility filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 1d ATR calculation (14-period)
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # ATR calculation
    atr_1d = np.zeros_like(tr)
    atr_1d[0] = tr[0]
    for i in range(1, len(tr)):
        atr_1d[i] = (atr_1d[i-1] * 13 + tr[i]) / 14
    
    # 1d ATR 50-period average for volatility ratio
    atr_ma_1d = pd.Series(atr_1d).rolling(window=50, min_periods=50).mean().values
    # Avoid division by zero
    atr_ratio_1d = np.where(atr_ma_1d > 0, atr_1d / atr_ma_1d, 1.0)
    
    # Align 1d ATR ratio to 6h timeframe
    atr_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio_1d)
    
    # Donchian channel (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume average (20-period)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(atr_ratio_1d_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 2 * ATR approximation using price range
            if close[i] < entry_price - 2.0 * (high[i] - low[i]):
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price breaks below Donchian low or volatility contracts
            elif close[i] < donchian_low[i] or atr_ratio_1d_aligned[i] < 0.8:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Stoploss: 2 * ATR approximation
            if close[i] > entry_price + 2.0 * (high[i] - low[i]):
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price breaks above Donchian high or volatility contracts
            elif close[i] > donchian_high[i] or atr_ratio_1d_aligned[i] < 0.8:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for breakout entries with volume and volatility confirmation
            # Long: price breaks above Donchian high, volatility expanding, volume spike
            if (close[i] > donchian_high[i] and 
                atr_ratio_1d_aligned[i] > 1.2 and
                volume[i] > 1.5 * volume_ma[i]):
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: price breaks below Donchian low, volatility expanding, volume spike
            elif (close[i] < donchian_low[i] and 
                  atr_ratio_1d_aligned[i] > 1.2 and
                  volume[i] > 1.5 * volume_ma[i]):
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Kaufman's Adaptive Moving Average (KAMA) crossover with 1d trend filter
# Long when KAMA_fast crosses above KAMA_slow, price > 1d EMA150 (uptrend), and volume > 1.3x average
# Short when KAMA_fast crosses below KAMA_slow, price < 1d EMA150 (downtrend), and volume > 1.3x average
# Uses KAMA for adaptive trend following that reduces whipsaws in choppy markets
# EMA150 filter ensures trades align with long-term trend
# Volume confirmation to validate signal strength
# ATR-based stoploss to limit drawdown (2x ATR)
# Target: 55-145 total trades over 4 years with controlled risk

name = "6h_kama_crossover_1d_ema150_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for EMA150 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 150:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # EMA150 calculation
    ema150_1d = pd.Series(close_1d).ewm(span=150, min_periods=150, adjust=False).mean().values
    
    # Align 1d EMA150 to 6h timeframe
    ema150_1d_aligned = align_htf_to_ltf(prices, df_1d, ema150_1d)
    
    # Kaufman's Adaptive Moving Average (KAMA)
    # Parameters: fast=2, slow=30
    def kama(close, fast=2, slow=30):
        # Calculate efficiency ratio
        change = np.abs(np.diff(close, n=10))  # 10-period change
        change = np.insert(change, 0, 0)  # Align with close
        
        volatility = np.zeros_like(close)
        for i in range(1, len(close)):
            volatility[i] = volatility[i-1] + np.abs(close[i] - close[i-1])
        
        er = np.zeros_like(close)
        for i in range(10, len(close)):
            if volatility[i] > 0:
                er[i] = change[i] / volatility[i]
            else:
                er[i] = 0
        
        # Smoothing constants
        fast_sc = 2 / (fast + 1)
        slow_sc = 2 / (slow + 1)
        
        # Calculate KAMA
        kama_values = np.zeros_like(close)
        kama_values[0] = close[0]