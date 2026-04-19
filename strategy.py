# 4h Donchian Breakout with Volume Confirmation and ATR Filter
# Breakout strategy using 20-period Donchian channels with volume confirmation (>2x average)
# and ATR-based volatility filter to avoid choppy markets. Long when price breaks above upper band,
# short when breaks below lower band. Uses 1d ATR for volatility filter to adapt to market conditions.
# Target: 20-50 trades/year per symbol to minimize fee drag.
# Works in both bull and breakout markets by capturing strong directional moves.

name = "4h_Donchian20_Volume_ATRFilter"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for ATR filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 20-period Donchian channels
    high_max_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 14-day ATR for volatility filter
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Align 1d ATR to 4h timeframe
    atr_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    
    # Volume confirmation: current volume > 2x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 14)  # Need Donchian and ATR data
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(high_max_20[i]) or np.isnan(low_min_20[i]) or 
            np.isnan(atr_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        upper_band = high_max_20[i]
        lower_band = low_min_20[i]
        atr_val = atr_aligned[i]
        vol_ma = vol_ma_20[i]
        vol = volume[i]
        
        # Volatility filter: only trade when ATR is above average (avoid choppy markets)
        # Using 50-period ATR average for normalization
        atr_ma_50 = pd.Series(atr_aligned).rolling(window=50, min_periods=50).mean().values
        if np.isnan(atr_ma_50[i]):
            volatility_filter = True  # Default to allowing trade if MA not ready
        else:
            volatility_filter = atr_val > 0.8 * atr_ma_50[i]  # Trade when volatility is not too low
        
        volume_confirmed = vol > 2.0 * vol_ma
        
        if position == 0:
            # Enter long: price breaks above upper Donchian band with volume and volatility
            if price > upper_band and volume_confirmed and volatility_filter:
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below lower Donchian band with volume and volatility
            elif price < lower_band and volume_confirmed and volatility_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long when price breaks below lower band or volatility drops significantly
            if price < lower_band or atr_val < 0.5 * atr_ma_50[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short when price breaks above upper band or volatility drops significantly
            if price > upper_band or atr_val < 0.5 * atr_ma_50[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals