# 12h_Camarilla_R1_S1_Breakout_Volume_ATRFilter_V1
# Hypothesis: 12h Camarilla pivot levels (R1/S1) act as strong support/resistance levels.
# Breakouts above R1 or below S1 with volume confirmation indicate institutional interest.
# ATR filter ensures we only trade when volatility is sufficient to avoid whipsaws in low volatility.
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag.
# Works in both bull and bear markets by following price action at key pivot levels.

name = "12h_Camarilla_R1_S1_Breakout_Volume_ATRFilter_V1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data ONCE before loop for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # === 1d Camarilla pivot levels (R1, S1) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivot point and Camarilla levels
    pivot = (high_1d + low_1d + close_1d) / 3.0
    r1 = close_1d + (high_1d - low_1d) * 1.1 / 12
    s1 = close_1d - (high_1d - low_1d) * 1.1 / 12
    
    # Align to 12h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # === 12h ATR for volatility filter ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # True Range calculation
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # ATR(14)
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # ATR ratio (current ATR / ATR MA) to filter low volatility
    atr_ma = pd.Series(atr).rolling(window=20, min_periods=20).mean().values
    atr_ratio = atr / np.where(atr_ma > 0, atr_ma, np.nan)
    
    # === 12h Volume confirmation ===
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(30, n):
        # Get values
        close_val = prices['close'].iloc[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        vol_ratio_val = vol_ratio[i]
        atr_ratio_val = atr_ratio[i]
        
        # Skip if any value is NaN
        if (np.isnan(r1_val) or np.isnan(s1_val) or np.isnan(vol_ratio_val) or np.isnan(atr_ratio_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Only trade when volatility is sufficient (ATR ratio > 0.8)
        vol_filter = atr_ratio_val > 0.8
        
        if position == 0:
            # Long: price breaks above R1 with volume confirmation and sufficient volatility
            if close_val > r1_val and vol_ratio_val > 1.3 and vol_filter:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
            # Short: price breaks below S1 with volume confirmation and sufficient volatility
            elif close_val < s1_val and vol_ratio_val > 1.3 and vol_filter:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
        
        elif position == 1:
            # Long exit: price breaks below S1 or volatility drops too low
            if close_val < s1_val or not vol_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above R1 or volatility drops too low
            if close_val > r1_val or not vol_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals