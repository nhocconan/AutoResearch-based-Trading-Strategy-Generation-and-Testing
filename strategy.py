#!/usr/bin/env python3
"""
Experiment #983: 6h Primary + 1d/1w HTF — Volume Spike + RSI Mean Reversion with Trend Filter

Hypothesis: 6h timeframe with volume spike detection + RSI extremes will capture 
panic bottoms and euphoria tops, while 1d/1w trend filter prevents counter-trend trades.
This combines proven mean-reversion edge (vol spike + RSI) with trend alignment.

Key innovations:
1. Volume Spike Detection: vol/avg_vol(20) > 2.0 signals capitulation/euphoria
2. RSI(14) Extremes: RSI<30 long, RSI>70 short (looser than typical 20/80 for more trades)
3. 1d HMA(21) Trend Filter: Only long if price>1d_HMA, only short if price<1d_HMA
4. 1w Momentum Bias: Weekly close>open = bull bias, prefer longs
5. CHOP(14) Regime: Adjust RSI thresholds based on trending vs ranging
6. ATR(14) 2.5x Trailing Stop: Protects from adverse moves

Why this should work:
- Volume spikes mark exhaustion points (panic selling = long opportunity)
- RSI extremes confirm oversold/overbought conditions
- 1d HMA filter prevents catching falling knives in strong downtrends
- 6h captures multi-day mean reversion swings
- Looser RSI thresholds (30/70 vs 20/80) ensure sufficient trade frequency

Entry conditions (LOOSE to guarantee trades):
- LONG = 1d bull + (vol_spike + RSI<35 OR CHOP>61 + RSI<40)
- SHORT = 1d bear + (vol_spike + RSI>65 OR CHOP>61 + RSI>60)
- Weekly momentum adds conviction but not required

Target: Sharpe>0.45, trades>=30 train, trades>=5 test, DD>-40%
Timeframe: 6h
Size: 0.25-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_volspike_rsi_trend_1d1w_v1"
timeframe = "6h"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average - responsive trend indicator"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = max(1, period // 2)
    sqrt_n = max(1, int(np.sqrt(period)))
    
    def wma(series, span):
        result = np.full(len(series), np.nan)
        weights = np.arange(1, span + 1, dtype=np.float64)
        for i in range(span - 1, len(series)):
            window = series[i - span + 1:i + 1].astype(np.float64)
            result[i] = np.sum(window * weights) / np.sum(weights)
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    diff = np.full(n, np.nan, dtype=np.float64)
    for i in range(period - 1, n):
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            diff[i] = 2.0 * wma_half[i] - wma_full[i]
    
    return wma(diff, sqrt_n)

def calculate_atr(high, low, close, period=14):
    """Average True Range - volatility measure for stops"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_rsi(close, period=14):
    """Relative Strength Index - momentum oscillator"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss != 0)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi[:period] = np.nan
    return rsi

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP)
    CHOP > 61.8 = range/choppy (mean reversion favorable)
    CHOP < 38.2 = trending (trend following favorable)
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    chop = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period, n):
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        
        tr_sum = 0.0
        for j in range(i-period+1, i+1):
            tr_sum += max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
        
        if highest_high > lowest_low and tr_sum > 1e-10:
            chop[i] = 100.0 * np.log10((highest_high - lowest_low) / tr_sum) / np.log10(period)
    
    return chop

def calculate_volume_zscore(volume, period=20):
    """Volume Z-Score - detects unusual volume spikes"""
    n = len(volume)
    if n < period + 1:
        return np.full(n, np.nan)
    
    vol_zscore = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period, n):
        window = volume[i-period+1:i+1]
        mean_vol = np.mean(window)
        std_vol = np.std(window)
        if std_vol > 1e-10:
            vol_zscore[i] = (volume[i] - mean_vol) / std_vol
        else:
            vol_zscore[i] = 0.0
    
    return vol_zscore

def calculate_volume_ratio(volume, period=20):
    """Volume Ratio - current vol vs average vol"""
    n = len(volume)
    if n < period + 1:
        return np.full(n, np.nan)
    
    vol_ratio = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period, n):
        avg_vol = np.mean(volume[i-period+1:i+1])
        if avg_vol > 1e-10:
            vol_ratio[i] = volume[i] / avg_vol
        else:
            vol_ratio[i] = 1.0
    
    return vol_ratio

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align HTF indicators
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Weekly momentum: close vs open
    weekly_momentum_raw = (df_1w['close'].values - df_1w['open'].values) / (df_1w['open'].values + 1e-10)
    weekly_momentum_aligned = align_htf_to_ltf(prices, df_1w, weekly_momentum_raw)
    
    # Calculate 6h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, period=14)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    vol_ratio = calculate_volume_ratio(volume, period=20)
    vol_zscore = calculate_volume_zscore(volume, period=20)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.30
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(rsi_14[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(weekly_momentum_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(chop_14[i]) or np.isnan(vol_ratio[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (1w momentum + 1d HMA) ===
        htf_1w_bull = weekly_momentum_aligned[i] > 0.0
        htf_1w_bear = weekly_momentum_aligned[i] < 0.0
        
        htf_1d_bull = close[i] > hma_1d_aligned[i]
        htf_1d_bear = close[i] < hma_1d_aligned[i]
        
        # === REGIME DETECTION (CHOP) ===
        is_trending = chop_14[i] < 38.2
        is_ranging = chop_14[i] > 61.8
        
        # === VOLUME SPIKE DETECTION ===
        vol_spike = vol_ratio[i] > 2.0  # Volume 2x average
        vol_extreme = vol_zscore[i] > 1.5  # Z-score > 1.5
        
        # === RSI EXTREMES (LOOSE THRESHOLDS FOR MORE TRADES) ===
        # Adjust thresholds based on regime
        if is_ranging:
            rsi_oversold = rsi_14[i] < 40  # Looser in range
            rsi_overbought = rsi_14[i] > 60
        else:
            rsi_oversold = rsi_14[i] < 35  # Tighter in trend
            rsi_overbought = rsi_14[i] > 65
        
        # === ENTRY LOGIC (VOLUME + RSI + TREND FILTER) ===
        desired_signal = 0.0
        
        # LONG entries - need 1d bullish bias
        if htf_1d_bull:
            # Strong long: volume spike + RSI oversold
            if vol_spike and rsi_oversold:
                desired_signal = SIZE_STRONG
            # Moderate long: ranging + RSI oversold
            elif is_ranging and rsi_14[i] < 45:
                desired_signal = SIZE_BASE
            # Continuation: 1w bull + pullback
            elif htf_1w_bull and rsi_14[i] < 50 and close[i] > hma_1d_aligned[i] * 0.98:
                desired_signal = SIZE_BASE
        
        # SHORT entries - need 1d bearish bias
        elif htf_1d_bear:
            # Strong short: volume spike + RSI overbought
            if vol_spike and rsi_overbought:
                desired_signal = -SIZE_STRONG
            # Moderate short: ranging + RSI overbought
            elif is_ranging and rsi_14[i] > 55:
                desired_signal = -SIZE_BASE
            # Continuation: 1w bear + rally
            elif htf_1w_bear and rsi_14[i] > 50 and close[i] < hma_1d_aligned[i] * 1.02:
                desired_signal = -SIZE_BASE
        
        # === STOPLOSS CHECK (2.5x ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.5 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.5 * entry_atr
            stop_price = min(stop_price, trailing_stop)
            if high[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= SIZE_STRONG * 0.9:
            final_signal = SIZE_STRONG
        elif desired_signal <= -SIZE_STRONG * 0.9:
            final_signal = -SIZE_STRONG
        elif desired_signal >= SIZE_BASE * 0.9:
            final_signal = SIZE_BASE
        elif desired_signal <= -SIZE_BASE * 0.9:
            final_signal = -SIZE_BASE
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position or np.sign(final_signal) != position_side:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                if position_side > 0:
                    stop_price = entry_price - 2.5 * entry_atr
                else:
                    stop_price = entry_price + 2.5 * entry_atr
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                stop_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = final_signal
    
    return signals