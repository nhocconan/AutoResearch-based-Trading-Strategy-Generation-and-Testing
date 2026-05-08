# 1h Candlestick Pattern + 4h Trend + Volume Confirmation
# Uses bullish/bearish engulfing patterns on 1h for entry timing
# 4h EMA50 for trend filter (avoid counter-trend trades)
# Volume spike confirms momentum
# Designed for 15-30 trades/year to minimize fee drag
# Works in bull/bear by using 4h trend filter

name = "1h_Engulfing_4hTrend_Volume"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    open_prices = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Volume spike: current volume > 2.0x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma20)
    
    # Bullish engulfing: current green candle engulfs previous red candle
    bullish_engulf = (close > open_prices) & (open_prices < close) & \
                     (close_prices_shift := np.roll(close, 1)) > (open_prices_shift := np.roll(open_prices, 1)) & \
                     (close > open_prices_shift) & (open_prices < close_prices_shift)
    # Bearish engulfing: current red candle engulfs previous green candle
    bearish_engulf = (close < open_prices) & (open_prices > close) & \
                     (close_prices_shift := np.roll(close, 1)) < (open_prices_shift := np.roll(open_prices, 1)) & \
                     (close < open_prices_shift) & (open_prices > open_prices_shift)
    
    # Fix first element for rolled arrays
    bullish_engulf[0] = False
    bearish_engulf[0] = False
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if np.isnan(ema_50_4h_aligned[i]) or np.isnan(volume_spike[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: bullish engulfing + 4h uptrend + volume spike
            long_cond = bullish_engulf[i] and ema_50_4h_aligned[i] > ema_50_4h_aligned[i-1] and volume_spike[i]
            
            # Short: bearish engulfing + 4h downtrend + volume spike
            short_cond = bearish_engulf[i] and ema_50_4h_aligned[i] < ema_50_4h_aligned[i-1] and volume_spike[i]
            
            if long_cond:
                signals[i] = 0.20
                position = 1
            elif short_cond:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: bearish engulfing or 4h trend breakdown
            if bearish_engulf[i] or ema_50_4h_aligned[i] < ema_50_4h_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: bullish engulfing or 4h trend breakdown
            if bullish_engulf[i] or ema_50_4h_aligned[i] > ema_50_4h_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals