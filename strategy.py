# 6h_Camarilla_R3S3_Breakout_1dTrend_VolumeSpike
# Uses Camarilla pivot levels from 1d to identify breakout zones.
# Long when price breaks above R3 with volume spike and above 1d EMA50 trend.
# Short when price breaks below S3 with volume spike and below 1d EMA50 trend.
# Exit when price returns to the 1d VWAP (mean reversion to daily average).
# Target: 50-150 total trades over 4 years (12-37/year) for low fee drift.

name = "6h_Camarilla_R3S3_Breakout_1dTrend_VolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for Camarilla pivots and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d Camarilla pivot levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla formula: Range = (high - low), Levels = close ± (Range * multiplier)
    range_1d = high_1d - low_1d
    # R3 = close + (range * 1.1/2) = close + range * 0.55
    # S3 = close - (range * 1.1/2) = close - range * 0.55
    r3_1d = close_1d + (range_1d * 0.55)
    s3_1d = close_1d - (range_1d * 0.55)
    
    # 1d VWAP for exit (mean reversion target)
    typical_price_1d = (high_1d + low_1d + close_1d) / 3.0
    vwap_1d = (typical_price_1d * df_1d['volume'].values).cumsum() / df_1d['volume'].values.cumsum()
    vwap_1d = vwap_1d  # already numpy array
    
    # 1d EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d indicators to 6h timeframe
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    vwap_1d_aligned = align_htf_to_ltf(prices, df_1d, vwap_1d)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # 6h volume filter: current volume > 2.0x 20-period average (strict to reduce trades)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (2.0 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(r3_1d_aligned[i]) or np.isnan(s3_1d_aligned[i]) or 
            np.isnan(vwap_1d_aligned[i]) or np.isnan(ema50_1d_aligned[i]) or
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: break above R3 with volume spike and above 1d EMA50
            long_break = close[i] > r3_1d_aligned[i]
            long_cond = long_break and volume_filter[i] and (close[i] > ema50_1d_aligned[i])
            
            # Short: break below S3 with volume spike and below 1d EMA50
            short_break = close[i] < s3_1d_aligned[i]
            short_cond = short_break and volume_filter[i] and (close[i] < ema50_1d_aligned[i])
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: return to 1d VWAP (mean reversion)
            if close[i] <= vwap_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: return to 1d VWAP (mean reversion)
            if close[i] >= vwap_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals