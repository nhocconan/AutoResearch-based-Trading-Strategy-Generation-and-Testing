#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot breakout with 1d volume confirmation and ATR-based volatility filter
# - Camarilla pivot levels on 12h: breakout above H3 = long, below L3 = short
# - Volume confirmation: current 1d volume > 1.5x 20-period EMA (avoid low-volume fakeouts)
# - Volatility filter: ATR(14) < 0.6 * 20-period ATR mean (avoid extreme volatility whipsaws)
# - Exit: opposite Camarilla breakout (H3/L3 reversal) or time-based exit (24h max hold = 2 bars)
# - Position sizing: 0.25 discrete level
# - Session filter: 08-20 UTC to avoid Asian session noise
# - Targets ~15-35 trades/year on 12h timeframe. Uses Camarilla for precise entry/exit,
#   volume confirmation reduces whipsaws, volatility filter adapts to market conditions.
#   Works in bull/bear: breakouts capture momentum, volume/volatility filters improve win rate.

name = "12h_1d_camarilla_volume_volatility_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Pre-compute HTF data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    bars_in_trade = 0  # track time in trade for max hold
    
    # Calculate Camarilla pivot levels on 12h (using previous bar's OHLC)
    # Camarilla: H4 = close + 1.1*(high-low)*1.1/2, H3 = close + 1.1*(high-low)*1.1/4
    #            L3 = close - 1.1*(high-low)*1.1/4, L4 = close - 1.1*(high-low)*1.1/2
    # Using previous bar to avoid look-ahead
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    prev_close[0] = close[0]
    
    range_hl = prev_high - prev_low
    camarilla_h3 = prev_close + 1.1 * range_hl * 1.1 / 4
    camarilla_l3 = prev_close - 1.1 * range_hl * 1.1 / 4
    
    # Calculate 1d volume EMA for confirmation
    volume_ema_20_1d = pd.Series(volume_1d).ewm(span=20, min_periods=20, adjust=False).mean().values
    volume_ema_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ema_20_1d)
    
    # Calculate ATR(14) for volatility filter
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first bar
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_mean_20 = pd.Series(atr_14).rolling(window=20, min_periods=20).mean().values
    
    # Pre-compute session filter (08-20 UTC)
    hours = prices.index.hour  # prices.index is DatetimeIndex
    in_session = (hours >= 8) & (hours <= 20)
    
    for i in range(100, n):
        # Skip if any required data is invalid or outside session
        if (np.isnan(camarilla_h3[i]) or np.isnan(camarilla_l3[i]) or 
            np.isnan(volume_ema_20_1d_aligned[i]) or np.isnan(atr_14[i]) or np.isnan(atr_mean_20[i]) or
            not in_session[i]):
            signals[i] = 0.0
            bars_in_trade = 0  # reset if forced flat
            continue
        
        # HTF volume confirmation: 1d volume > 1.5x its 20-period EMA
        vol_1d_current = align_htf_to_ltf(prices, df_1d, volume_1d)
        vol_confirm_1d = vol_1d_current[i] > 1.5 * volume_ema_20_1d_aligned[i]
        
        # Volatility filter: ATR(14) < 0.6 * 20-period ATR mean (avoid extreme volatility)
        vol_filter = atr_14[i] < 0.6 * atr_mean_20[i]
        
        # Entry conditions
        long_entry = (close[i] > camarilla_h3[i] and 
                     vol_confirm_1d and 
                     vol_filter)
        short_entry = (close[i] < camarilla_l3[i] and 
                      vol_confirm_1d and 
                      vol_filter)
        
        if position == 0:  # Flat - look for entry
            if long_entry:
                position = 1
                signals[i] = 0.25
                bars_in_trade = 0
            elif short_entry:
                position = -1
                signals[i] = -0.25
                bars_in_trade = 0
            else:
                signals[i] = 0.0
                bars_in_trade = 0
        else:  # Have position - look for exit
            bars_in_trade += 1
            # Exit conditions: opposite Camarilla breakout or max hold (2 bars = 24h)
            if position == 1:  # Long position
                if (close[i] < camarilla_l3[i] or  # opposite breakout
                    bars_in_trade >= 2):           # max hold time
                    position = 0
                    signals[i] = 0.0
                    bars_in_trade = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1 (Short position)
                if (close[i] > camarilla_h3[i] or  # opposite breakout
                    bars_in_trade >= 2):           # max hold time
                    position = 0
                    signals[i] = 0.0
                    bars_in_trade = 0
                else:
                    signals[i] = -0.25
    
    return signals