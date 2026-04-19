# 12h_Camarilla_R1_S1_Breakout_Volume_ATRFilter_V1
# Hypothesis: 12h Camarilla pivot breakout with volume confirmation and ATR-based volatility filter.
# Long when price breaks above R1 with volume > 1.5x 12h average and ATR ratio < 0.8.
# Short when price breaks below S1 with volume > 1.5x 12h average and ATR ratio < 0.8.
# Exit when price returns to pivot point or volatility increases (ATR ratio > 1.2).
# Uses 1d Camarilla levels for structure, volume for confirmation, ATR for volatility regime.
# Designed for 12-30 trades/year per symbol to minimize fee drag.

name = "12h_Camarilla_R1_S1_Breakout_Volume_ATRFilter_V1"
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
    
    # 1d data for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels for each 1d bar
    # Pivot = (H + L + C) / 3
    # R1 = C + (H - L) * 1.1 / 12
    # S1 = C - (H - L) * 1.1 / 12
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    r1_1d = close_1d + (high_1d - low_1d) * 1.1 / 12.0
    s1_1d = close_1d - (high_1d - low_1d) * 1.1 / 12.0
    
    # Align Camarilla levels to 12h timeframe (wait for 1d bar to close)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # 12h volume average (20-period)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # 1d ATR for volatility filter
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_1d = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_ma_10 = pd.Series(atr_1d).rolling(window=10, min_periods=10).mean().values
    atr_ratio = atr_1d / (atr_ma_10 + 1e-10)
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(volume_ma[i]) or np.isnan(atr_ratio_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = volume_ma[i]
        pivot = pivot_aligned[i]
        r1 = r1_aligned[i]
        s1 = s1_aligned[i]
        atr_ratio_val = atr_ratio_aligned[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirm = vol > 1.5 * vol_ma
        
        # Volatility filter: ATR ratio < 0.8 (low volatility environment)
        vol_filter = atr_ratio_val < 0.8
        
        if position == 0:
            # Long: price breaks above R1 with volume confirmation and low volatility
            if price > r1 and volume_confirm and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with volume confirmation and low volatility
            elif price < s1 and volume_confirm and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price returns to pivot or volatility increases
            if price < pivot or atr_ratio_val > 1.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns to pivot or volatility increases
            if price > pivot or atr_ratio_val > 1.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals