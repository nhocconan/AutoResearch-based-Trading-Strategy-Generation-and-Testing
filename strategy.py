# 12h 1w EMA Trend with 1d Volume Confirmation and ATR Stop
# Hypothesis: Weekly EMA trend filter on 12h timeframe with daily volume confirmation and ATR-based stops.
# The 1w EMA provides strong trend filtering that works in both bull and bear markets.
# Volume spike confirms institutional interest, reducing false breakouts.
# ATR stops manage risk during volatile periods. Target: 15-35 trades/year.

name = "12h_1w_EMA_Trend_1d_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get weekly data ONCE before loop for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Get daily data ONCE before loop for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 20-period EMA on weekly close
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align weekly EMA to 12h timeframe (waits for weekly close)
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Calculate 20-period average volume on daily data
    volume_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align daily volume MA to 12h timeframe (waits for daily close)
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # Calculate 14-period ATR on 12h data for stop loss
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    # Wilder's smoothing for ATR
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan, dtype=float)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.nansum(data[:period])
        # Subsequent values
        for i in range(period, len(data)):
            result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    atr = wilders_smoothing(tr, 14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after warmup
        # Get values
        close_val = prices['close'].iloc[i]
        ema_val = ema_20_1w_aligned[i]
        vol_ma_val = vol_ma_20_1d_aligned[i]
        vol_current = volume_1d[-1] if len(volume_1d) > 0 else 0  # Current daily volume
        
        # Get current daily volume (need to align to current 12h bar)
        # Find the most recent daily volume that's available
        if i < len(prices):
            # Use the volume from the most recent completed day
            day_idx = min(i // 2, len(df_1d) - 1)  # Approximate: 2x 12h bars per day
            if day_idx < len(volume_1d):
                vol_current = volume_1d[day_idx]
            else:
                vol_current = volume_1d[-1] if len(volume_1d) > 0 else 0
        
        # Skip if any value is NaN
        if (np.isnan(ema_val) or np.isnan(vol_ma_val) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: price above weekly EMA, volume > 1.5x average
            if close_val > ema_val and vol_current > vol_ma_val * 1.5:
                signals[i] = 0.25
                position = 1
            # Short entry: price below weekly EMA, volume > 1.5x average
            elif close_val < ema_val and vol_current > vol_ma_val * 1.5:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below weekly EMA OR ATR-based stop
            if close_val <= ema_val or close_val <= prices['high'].iloc[i] - 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above weekly EMA OR ATR-based stop
            if close_val >= ema_val or close_val >= prices['low'].iloc[i] + 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals