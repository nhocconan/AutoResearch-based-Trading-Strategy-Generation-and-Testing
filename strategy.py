# 6h_Camarilla_R3S3_Breakout_1dTrend_VolumeSpike_HT
# Strategy: Camarilla pivot breakout with daily trend filter and volume confirmation
# Long when price breaks above R3 (1d Camarilla) AND 1d EMA34 > EMA89 (uptrend) AND 6h volume > 1.5x 20-period average
# Short when price breaks below S3 (1d Camarilla) AND 1d EMA34 < EMA89 (downtrend) AND 6h volume > 1.5x 20-period average
# Exit when price crosses back to daily pivot point (mean reversion to equilibrium)
# Uses proven Camarilla structure with trend filter to avoid false breakouts in ranging markets
# Target: 50-150 total trades over 4 years (12-37/year) for low fee drag
# Works in bull (breakouts continue) and bear (mean reversion to pivot)

name = "6h_Camarilla_R3S3_Breakout_1dTrend_VolumeSpike_HT"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 6h volume filter: current volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma20)
    
    # 1d data for Camarilla levels and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day's OHLC
    # Using previous day's data to avoid look-ahead
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    open_1d = df_1d['open'].values
    
    # Previous day's typical price (Camarilla uses previous day's range)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    
    # Handle first value
    prev_high[0] = high_1d[0]
    prev_low[0] = low_1d[0]
    prev_close[0] = close_1d[0]
    
    # Camarilla levels: based on previous day's range
    range_ = prev_high - prev_low
    camarilla_pivot = (prev_high + prev_low + prev_close) / 3
    camarilla_r3 = camarilla_pivot + (1.1 * range_ / 2)
    camarilla_s3 = camarilla_pivot - (1.1 * range_ / 2)
    
    # 1d EMA trend filter: EMA34 vs EMA89
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_89 = pd.Series(close_1d).ewm(span=89, adjust=False, min_periods=89).mean().values
    uptrend = ema_34 > ema_89
    downtrend = ema_34 < ema_89
    
    # Align 1d indicators to 6h timeframe
    camarilla_pivot_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pivot)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    uptrend_aligned = align_htf_to_ltf(prices, df_1d, uptrend)
    downtrend_aligned = align_htf_to_ltf(prices, df_1d, downtrend)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 90  # Sufficient warmup for EMA89
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(camarilla_pivot_aligned[i]) or np.isnan(uptrend_aligned[i]) or 
            np.isnan(downtrend_aligned[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: break above R3, uptrend, volume spike
            long_cond = (close[i] > camarilla_r3_aligned[i]) and uptrend_aligned[i] and volume_filter[i]
            # Short conditions: break below S3, downtrend, volume spike
            short_cond = (close[i] < camarilla_s3_aligned[i]) and downtrend_aligned[i] and volume_filter[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: return to pivot (mean reversion)
            if close[i] < camarilla_pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: return to pivot (mean reversion)
            if close[i] > camarilla_pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals