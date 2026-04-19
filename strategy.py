# 4h RSI Divergence with Volume Confirmation and ADX Trend Filter
# Hypothesis: RSI divergence signals exhaustion, volume confirms momentum shift, ADX filters for trending conditions.
# Works in both bull and bear by capturing reversals at extremes. Target: 20-30 trades/year.
name = "4h_RSIDiv_Volume_ADX"
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
    
    # Get 1d RSI for divergence and 14-period ADX for trend strength
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 14-period RSI
    delta = pd.Series(df_1d['close']).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # Calculate 14-period ADX
    plus_dm = pd.Series(df_1d['high']).diff()
    minus_dm = pd.Series(df_1d['low']).diff()
    plus_dm = plus_dm.where((plus_dm > 0) & (plus_dm > -minus_dm), 0.0)
    minus_dm = minus_dm.where((-minus_dm > 0) & (-minus_dm > plus_dm), 0.0)
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = np.abs(df_1d['high'] - df_1d['close'].shift(1))
    tr3 = np.abs(df_1d['low'] - df_1d['close'].shift(1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1/14, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(alpha=1/14, adjust=False).mean() / atr)
    dx = (np.abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    adx = dx.ewm(alpha=1/14, adjust=False).mean()
    adx_values = adx.values
    
    # Get 1d average volume for confirmation
    vol_ma_1d = pd.Series(df_1d['volume']).rolling(window=20, min_periods=20).mean().values
    
    # Align 1d indicators to 4h timeframe
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi_values)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx_values)
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Calculate 4-hour RSI for entry timing
    delta_4h = pd.Series(close).diff()
    gain_4h = delta_4h.clip(lower=0)
    loss_4h = -delta_4h.clip(upper=0)
    avg_gain_4h = gain_4h.ewm(alpha=1/14, adjust=False).mean()
    avg_loss_4h = loss_4h.ewm(alpha=1/14, adjust=False).mean()
    rs_4h = avg_gain_4h / avg_loss_4h
    rsi_4h = 100 - (100 / (1 + rs_4h))
    rsi_4h_values = rsi_4h.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 50)  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(rsi_aligned[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(vol_ma_1d_aligned[i]) or np.isnan(rsi_4h_values[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        rsi_1d = rsi_aligned[i]
        adx_val = adx_aligned[i]
        vol_ma = vol_ma_1d_aligned[i]
        vol = volume[i]
        rsi_4h = rsi_4h_values[i]
        
        # Trend filter: only trade when ADX > 25 (trending market)
        trend_filter = adx_val > 25
        
        # Volume confirmation: volume > 1.5x daily average
        vol_confirm = vol > 1.5 * vol_ma
        
        # RSI divergence signals (simplified: look for RSI extremes with 4h confirmation)
        if position == 0:
            # Bullish divergence setup: 1d RSI oversold (<30) + 4h RSI turning up (>40 from below) + volume + trend
            if (rsi_1d < 30 and rsi_4h > 40 and 
                i > start_idx and rsi_4h_values[i-1] <= 40 and 
                vol_confirm and trend_filter):
                signals[i] = 0.25
                position = 1
            # Bearish divergence setup: 1d RSI overbought (>70) + 4h RSI turning down (<60 from above) + volume + trend
            elif (rsi_1d > 70 and rsi_4h < 60 and 
                  i > start_idx and rsi_4h_values[i-1] >= 60 and 
                  vol_confirm and trend_filter):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: 4h RSI overbought (>70) or 1d RSI overbought
            if rsi_4h > 70 or rsi_1d > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: 4h RSI oversold (<30) or 1d RSI oversold
            if rsi_4h < 30 or rsi_1d < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals