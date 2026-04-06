#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h mean reversion with 4h trend filter and 1d volatility filter
# Long when price pulls back to 4h VWAP during 1d low volatility regime
# Short when price rallies to 4h VWAP during 1d low volatility regime
# Uses 1h RSI for entry timing and 4h trend for direction
# Target: 80-120 total trades over 4 years with controlled risk
# Volatility filter prevents trades during high volatility periods

name = "1h_vwap_rsi_4h_trend_1d_vol_v1"
timeframe = "1h"
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
    
    # 4h data for VWAP and trend
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    volume_4h = df_4h['volume'].values
    
    # Calculate 4h VWAP (typical price * volume)
    typical_price_4h = (high_4h + low_4h + close_4h) / 3.0
    vwap_4h = (typical_price_4h * volume_4h).cumsum() / volume_4h.cumsum()
    vwap_4h = vwap_4h.astype(float)  # ensure float type
    
    # 4h EMA20 for trend filter
    ema20_4h = pd.Series(close_4h).ewm(span=20, min_periods=20, adjust=False).mean().values
    
    # 1d data for volatility filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # 1d ATR(14) for volatility
    def calculate_atr(high, low, close, period=14):
        tr1 = np.abs(high[1:] - low[1:])
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr = np.concatenate([[np.nan], tr])
        atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
        return atr
    
    atr_1d = calculate_atr(high_1d, low_1d, close_1d, 14)
    
    # 1d ATR percentage of price (volatility regime)
    atr_pct_1d = atr_1d / close_1d
    
    # Align 4h indicators to 1h timeframe
    vwap_4h_aligned = align_htf_to_ltf(prices, df_4h, vwap_4h)
    ema20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema20_4h)
    atr_pct_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_pct_1d)
    
    # 1h RSI(14) for entry timing
    def calculate_rsi(close, period=14):
        delta = np.diff(close)
        delta = np.concatenate([[np.nan], delta])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        gain = pd.Series(gain).ewm(alpha=1/period, min_periods=period, adjust=False).mean().values
        loss = pd.Series(loss).ewm(alpha=1/period, min_periods=period, adjust=False).mean().values
        rs = gain / (loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    rsi_1h = calculate_rsi(close, 14)
    
    # 1h volatility filter (avoid trading during high volatility)
    def calculate_atr_1h(high, low, close, period=14):
        tr1 = np.abs(high[1:] - low[1:])
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr = np.concatenate([[np.nan], tr])
        atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
        return atr
    
    atr_1h = calculate_atr_1h(high, low, close, 14)
    atr_pct_1h = atr_1h / close
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if required data not available
        if (np.isnan(vwap_4h_aligned[i]) or np.isnan(ema20_4h_aligned[i]) or 
            np.isnan(atr_pct_1d_aligned[i]) or np.isnan(rsi_1h[i]) or
            np.isnan(atr_pct_1h[i])):
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        # Volatility filter: only trade when 1d volatility is low (< 0.02 = 2%)
        vol_filter = atr_pct_1d_aligned[i] < 0.02
        
        if position == 1:  # long position
            # Stoploss: 2 * ATR
            if close[i] < entry_price - 2.0 * atr_1h[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price moves above VWAP or RSI overbought
            elif close[i] > vwap_4h_aligned[i] or rsi_1h[i] > 70:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.20
        elif position == -1:  # short position
            # Stoploss: 2 * ATR
            if close[i] > entry_price + 2.0 * atr_1h[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price moves below VWAP or RSI oversold
            elif close[i] < vwap_4h_aligned[i] or rsi_1h[i] < 30:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.20
        else:
            # Look for mean reversion entries with volatility filter
            # Long: price pulls back to VWAP in uptrend + RSI oversold + low vol
            if (close[i] <= vwap_4h_aligned[i] * 1.001 and  # allow small tolerance
                close[i] >= vwap_4h_aligned[i] * 0.999 and
                ema20_4h_aligned[i] > ema20_4h_aligned[max(0, i-1)] and  # 4h uptrend
                rsi_1h[i] < 30 and
                vol_filter):
                signals[i] = 0.20
                position = 1
                entry_price = close[i]
            # Short: price rallies to VWAP in downtrend + RSI overbought + low vol
            elif (close[i] >= vwap_4h_aligned[i] * 0.999 and  # allow small tolerance
                  close[i] <= vwap_4h_aligned[i] * 1.001 and
                  ema20_4h_aligned[i] < ema20_4h_aligned[max(0, i-1)] and  # 4h downtrend
                  rsi_1h[i] > 70 and
                  vol_filter):
                signals[i] = -0.20
                position = -1
                entry_price = close[i]
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h mean reversion with 4h trend filter and 1d volatility filter
# Long when price pulls back to 4h VWAP during 1d low volatility regime
# Short when price rallies to 4h VWAP during 1d low volatility regime
# Uses 1h RSI for entry timing and 4h trend for direction
# Target: 80-120 total trades over 4 years with controlled risk
# Volatility filter prevents trades during high volatility periods

name = "1h_vwap_rsi_4h_trend_1d_vol_v1"
timeframe = "1h"
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
    
    # 4h data for VWAP and trend
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    volume_4h = df_4h['volume'].values
    
    # Calculate 4h VWAP (typical price * volume)
    typical_price_4h = (high_4h + low_4h + close_4h) / 3.0
    vwap_4h = (typical_price_4h * volume_4h).cumsum() / volume_4h.cumsum()
    vwap_4h = vwap_4h.astype(float)  # ensure float type
    
    # 4h EMA20 for trend filter
    ema20_4h = pd.Series(close_4h).ewm(span=20, min_periods=20, adjust=False).mean().values
    
    # 1d data for volatility filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # 1d ATR(14) for volatility
    def calculate_atr(high, low, close, period=14):
        tr1 = np.abs(high[1:] - low[1:])
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr = np.concatenate([[np.nan], tr])
        atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
        return atr
    
    atr_1d = calculate_atr(high_1d, low_1d, close_1d, 14)
    
    # 1d ATR percentage of price (volatility regime)
    atr_pct_1d = atr_1d / close_1d
    
    # Align 4h indicators to 1h timeframe
    vwap_4h_aligned = align_htf_to_ltf(prices, df_4h, vwap_4h)
    ema20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema20_4h)
    atr_pct_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_pct_1d)
    
    # 1h RSI(14) for entry timing
    def calculate_rsi(close, period=14):
        delta = np.diff(close)
        delta = np.concatenate([[np.nan], delta])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        gain = pd.Series(gain).ewm(alpha=1/period, min_periods=period, adjust=False).mean().values
        loss = pd.Series(loss).ewm(alpha=1/period, min_periods=period, adjust=False).mean().values
        rs = gain / (loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    rsi_1h = calculate_rsi(close, 14)
    
    # 1h volatility filter (avoid trading during high volatility)
    def calculate_atr_1h(high, low, close, period=14):
        tr1 = np.abs(high[1:] - low[1:])
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr = np.concatenate([[np.nan], tr])
        atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
        return atr
    
    atr_1h = calculate_atr_1h(high, low, close, 14)
    atr_pct_1h = atr_1h / close
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if required data not available
        if (np.isnan(vwap_4h_aligned[i]) or np.isnan(ema20_4h_aligned[i]) or 
            np.isnan(atr_pct_1d_aligned[i]) or np.isnan(rsi_1h[i]) or
            np.isnan(atr_pct_1h[i])):
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        # Volatility filter: only trade when 1d volatility is low (< 0.02 = 2%)
        vol_filter = atr_pct_1d_aligned[i] < 0.02
        
        if position == 1:  # long position
            # Stoploss: 2 * ATR
            if close[i] < entry_price - 2.0 * atr_1h[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price moves above VWAP or RSI overbought
            elif close[i] > vwap_4h_aligned[i] or rsi_1h[i] > 70:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.20
        elif position == -1:  # short position
            # Stoploss: 2 * ATR
            if close[i] > entry_price + 2.0 * atr_1h[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price moves below VWAP or RSI oversold
            elif close[i] < vwap_4h_aligned[i] or rsi_1h[i] < 30:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.20
        else:
            # Look for mean reversion entries with volatility filter
            # Long: price pulls back to VWAP in uptrend + RSI oversold + low vol
            if (close[i] <= vwap_4h_aligned[i] * 1.001 and  # allow small tolerance
                close[i] >= vwap_4h_aligned[i] * 0.999 and
                ema20_4h_aligned[i] > ema20_4h_aligned[max(0, i-1)] and  # 4h uptrend
                rsi_1h[i] < 30 and
                vol_filter):
                signals[i] = 0.20
                position = 1
                entry_price = close[i]
            # Short: price rallies to VWAP in downtrend + RSI overbought + low vol
            elif (close[i] >= vwap_4h_aligned[i] * 0.999 and  # allow small tolerance
                  close[i] <= vwap_4h_aligned[i] * 1.001 and
                  ema20_4h_aligned[i] < ema20_4h_aligned[max(0, i-1)] and  # 4h downtrend
                  rsi_1h[i] > 70 and
                  vol_filter):
                signals[i] = -0.20
                position = -1
                entry_price = close[i]
    
    return signals