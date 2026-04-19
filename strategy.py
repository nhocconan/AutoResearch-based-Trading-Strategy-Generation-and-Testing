# 12h_1dPivot_R1S1_Breakout_Volume_12hEMA34
# Hypothesis: Price breaking above/below daily Camarilla pivot levels (R1/S1) with volume confirmation and 12h EMA trend filter
# Works in bull markets (breakouts continue) and bear markets (reversions at S1/R1)
# Pivot levels provide institutional support/resistance, volume confirms participation, EMA filter avoids counter-trend trades
# Target: 15-30 trades/year per symbol (~60-120 total over 4 years)

name = "12h_1dPivot_R1S1_Breakout_Volume_12hEMA34"
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
    
    # Get daily data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 12h EMA34 for trend filter
    ema_34_12h = pd.Series(close).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate Camarilla pivot levels from previous day
    # R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    range_1d = high_1d - low_1d
    r1 = close_1d + range_1d * 1.1 / 12
    s1 = close_1d - range_1d * 1.1 / 12
    
    # Align pivot levels to 12h timeframe (use previous day's levels)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)  # Need EMA and volume MA data
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema_34_12h[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        ema_trend = ema_34_12h[i]
        r1_level = r1_aligned[i]
        s1_level = s1_aligned[i]
        vol_ma = vol_ma_20[i]
        vol = volume[i]
        
        # Volume threshold
        volume_confirmed = vol > 1.5 * vol_ma
        
        if position == 0:
            # Enter long: price breaks above R1 with volume and above EMA trend
            if price > r1_level and volume_confirmed and price > ema_trend:
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below S1 with volume and below EMA trend
            elif price < s1_level and volume_confirmed and price < ema_trend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price falls below S1 or below EMA trend
            if price < s1_level or price < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price rises above R1 or above EMA trend
            if price > r1_level or price > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals