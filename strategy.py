# 12h_Median_CCI_D1_1wTrend_v1
# Hypothesis: 12h median CCI(20) with 1d volume confirmation and 1w trend filter
# - Median CCI(20) uses median price (HLC/3) to reduce noise and improve robustness
# - Long when median CCI crosses above -100 (emerging uptrend) with volume confirmation
# - Short when median CCI crosses below +100 (emerging downtrend) with volume confirmation
# - 1w EMA(34) trend filter ensures alignment with weekly trend direction
# - Volume filter: 12h volume > 1.3x 20-period average for institutional participation
# - Designed to work in both bull and bear markets by following higher timeframe trend
# - Target: 15-30 trades/year to avoid excessive fee drag

name = "12h_Median_CCI_D1_1wTrend_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    
    # 1d volume average (20-period)
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # 1w EMA(34) for trend direction
    ema_34_1w = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Median price for CCI calculation (typical price = (H+L+C)/3)
    median_price = (high + low + close) / 3.0
    
    # CCI(20) calculation
    # CCI = (Typical Price - SMA) / (0.015 * Mean Deviation)
    tp = median_price
    sma_20 = pd.Series(tp).rolling(window=20, min_periods=20).mean().values
    
    # Mean deviation calculation
    def mean_deviation(arr, window):
        md = np.zeros_like(arr)
        for i in range(window-1, len(arr)):
            md[i] = np.mean(np.abs(arr[i-window+1:i+1] - np.mean(arr[i-window+1:i+1])))
        return md
    
    md_20 = mean_deviation(tp, 20)
    cci = (tp - sma_20) / (0.015 * md_20 + 1e-10)  # Add small epsilon to avoid division by zero
    
    # Pre-compute session filter (00:00-23:00 UTC - trade all hours for 12h)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    # For 12h timeframe, trade during active market hours (00:00-23:00 UTC covers all)
    in_session = np.ones(n, dtype=bool)  # Trade all hours for 12h
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(cci[i]) or np.isnan(sma_20[i]) or np.isnan(md_20[i]) or 
            np.isnan(vol_ma_1d_aligned[i]) or np.isnan(ema_34_1w_aligned[i])):
            signals[i] = 0.0
            continue
            
        # Volume filter: current 12h volume > 1.3x average
        volume_filter = vol_ma_1d_aligned[i] > 0 and volume[i] > 1.3 * vol_ma_1d_aligned[i]
        
        # Trend filter: align with 1w EMA direction
        uptrend = close[i] > ema_34_1w_aligned[i]
        downtrend = close[i] < ema_34_1w_aligned[i]
        
        if position == 0:
            # Look for long entry: CCI crosses above -100 + uptrend + volume
            if cci[i] > -100 and cci[i-1] <= -100 and uptrend and volume_filter:
                signals[i] = 0.25
                position = 1
            # Look for short entry: CCI crosses below +100 + downtrend + volume
            elif cci[i] < 100 and cci[i-1] >= 100 and downtrend and volume_filter:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long position: exit on CCI crossing below +100 or trend reversal
            if cci[i] < 100 and cci[i-1] >= 100 or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short position: exit on CCI crossing above -100 or trend reversal
            if cci[i] > -100 and cci[i-1] <= -100 or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals