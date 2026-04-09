#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h EMA pullback strategy with 4h trend filter and 1d volume regime
# - Uses 4h EMA(50) for trend direction (long when price > EMA, short when price < EMA)
# - Enters on 1h pullbacks to EMA(21) with RSI(14) < 40 for longs or > 60 for shorts
# - Uses 1d ATR(14) normalized volume spike (>1.5x 20-period average) for confirmation
# - Includes session filter (08-20 UTC) to avoid low-liquidity hours
# - Fixed position size 0.20 to control drawdown
# - Target: 15-35 trades/year (60-140 total over 4 years) on 1h timeframe

name = "1h_4h_1d_ema_pullback_volume_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_4h) < 60 or len(df_1d) < 60:
        return np.zeros(n)
    
    # Pre-compute 4h indicators
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # 4h EMA(50) for trend filter
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Pre-compute 1d indicators
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # 1d ATR(14) for volatility normalization
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    tr_1d[0] = tr_1d[0]
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    
    # 1d Volume > 1.5x 20-period average (volume regime filter)
    avg_volume_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_regime_1d = volume_1d > (1.5 * avg_volume_20)
    
    # Align 1d indicators to 1h
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    volume_regime_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_regime_1d.astype(float))
    
    # 1h price data
    open_1h = prices['open'].values
    high_1h = prices['high'].values
    low_1h = prices['low'].values
    close_1h = prices['close'].values
    volume_1h = prices['volume'].values
    
    # 1h EMA(21) for pullback entries
    ema_21_1h = pd.Series(close_1h).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # 1h RSI(14) for momentum confirmation
    delta = pd.Series(close_1h).diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=14).mean()
    rs = gain / loss
    rsi_14_1h = 100 - (100 / (1 + rs))
    rsi_14_1h = rsi_14_1h.fillna(50).values
    
    # Session filter: 08-20 UTC (pre-compute for efficiency)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(ema_21_1h[i]) or
            np.isnan(rsi_14_1h[i]) or np.isnan(atr_1d_aligned[i]) or
            np.isnan(volume_regime_1d_aligned[i]) or atr_1d_aligned[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Session filter
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for entries
            # Long conditions: 4h uptrend + 1h pullback to EMA(21) + oversold RSI + volume regime
            if (close_1h[i] > ema_50_4h_aligned[i] and    # 4h uptrend
                low_1h[i] <= ema_21_1h[i] and             # Pullback to 1h EMA(21)
                rsi_14_1h[i] < 40 and                     # Oversold RSI
                volume_regime_1d_aligned[i]):             # High volume regime
                position = 1
                signals[i] = 0.20
            # Short conditions: 4h downtrend + 1h pullback to EMA(21) + overbought RSI + volume regime
            elif (close_1h[i] < ema_50_4h_aligned[i] and  # 4h downtrend
                  high_1h[i] >= ema_21_1h[i] and          # Pullback to 1h EMA(21)
                  rsi_14_1h[i] > 60 and                   # Overbought RSI
                  volume_regime_1d_aligned[i]):           # High volume regime
                position = -1
                signals[i] = -0.20
        elif position == 1:  # Long position - exit on 4h trend reversal or RSI overbought
            if (close_1h[i] < ema_50_4h_aligned[i] or   # 4h trend reversal
                rsi_14_1h[i] > 70):                     # Overbought exit
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
        elif position == -1:  # Short position - exit on 4h trend reversal or RSI oversold
            if (close_1h[i] > ema_50_4h_aligned[i] or   # 4h trend reversal
                rsi_14_1h[i] < 30):                     # Oversold exit
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
    
    return signals

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h EMA pullback strategy with 4h trend filter and 1d volume regime
# - Uses 4h EMA(50) for trend direction (long when price > EMA, short when price < EMA)
# - Enters on 1h pullbacks to EMA(21) with RSI(14) < 40 for longs or > 60 for shorts
# - Uses 1d ATR(14) normalized volume spike (>1.5x 20-period average) for confirmation
# - Includes session filter (08-20 UTC) to avoid low-liquidity hours
# - Fixed position size 0.20 to control drawdown
# - Target: 15-35 trades/year (60-140 total over 4 years) on 1h timeframe

name = "1h_4h_1d_ema_pullback_volume_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_4h) < 60 or len(df_1d) < 60:
        return np.zeros(n)
    
    # Pre-compute 4h indicators
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # 4h EMA(50) for trend filter
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Pre-compute 1d indicators
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # 1d ATR(14) for volatility normalization
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    tr_1d[0] = tr_1d[0]
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    
    # 1d Volume > 1.5x 20-period average (volume regime filter)
    avg_volume_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_regime_1d = volume_1d > (1.5 * avg_volume_20)
    
    # Align 1d indicators to 1h
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    volume_regime_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_regime_1d.astype(float))
    
    # 1h price data
    open_1h = prices['open'].values
    high_1h = prices['high'].values
    low_1h = prices['low'].values
    close_1h = prices['close'].values
    volume_1h = prices['volume'].values
    
    # 1h EMA(21) for pullback entries
    ema_21_1h = pd.Series(close_1h).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # 1h RSI(14) for momentum confirmation
    delta = pd.Series(close_1h).diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=14).mean()
    rs = gain / loss
    rsi_14_1h = 100 - (100 / (1 + rs))
    rsi_14_1h = rsi_14_1h.fillna(50).values
    
    # Session filter: 08-20 UTC (pre-compute for efficiency)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(ema_21_1h[i]) or
            np.isnan(rsi_14_1h[i]) or np.isnan(atr_1d_aligned[i]) or
            np.isnan(volume_regime_1d_aligned[i]) or atr_1d_aligned[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Session filter
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for entries
            # Long conditions: 4h uptrend + 1h pullback to EMA(21) + oversold RSI + volume regime
            if (close_1h[i] > ema_50_4h_aligned[i] and    # 4h uptrend
                low_1h[i] <= ema_21_1h[i] and             # Pullback to 1h EMA(21)
                rsi_14_1h[i] < 40 and                     # Oversold RSI
                volume_regime_1d_aligned[i]):             # High volume regime
                position = 1
                signals[i] = 0.20
            # Short conditions: 4h downtrend + 1h pullback to EMA(21) + overbought RSI + volume regime
            elif (close_1h[i] < ema_50_4h_aligned[i] and  # 4h downtrend
                  high_1h[i] >= ema_21_1h[i] and          # Pullback to 1h EMA(21)
                  rsi_14_1h[i] > 60 and                   # Overbought RSI
                  volume_regime_1d_aligned[i]):           # High volume regime
                position = -1
                signals[i] = -0.20
        elif position == 1:  # Long position - exit on 4h trend reversal or RSI overbought
            if (close_1h[i] < ema_50_4h_aligned[i] or   # 4h trend reversal
                rsi_14_1h[i] > 70):                     # Overbought exit
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
        elif position == -1:  # Short position - exit on 4h trend reversal or RSI oversold
            if (close_1h[i] > ema_50_4h_aligned[i] or   # 4h trend reversal
                rsi_14_1h[i] < 30):                     # Oversold exit
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
    
    return signals