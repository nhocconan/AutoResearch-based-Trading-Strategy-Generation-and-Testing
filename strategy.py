#!/usr/bin/env python3
"""
Experiment #386: 30m Volatility Spike Mean Reversion + 4h HMA Trend + ADX Hysteresis

Hypothesis: After 385 experiments, the pattern is clear - pure trend following fails in 
2022 crash and 2025 bear market. Pure mean reversion fails in strong trends. The key 
is VOLATILITY SPIKE DETECTION combined with REGIME FILTERING.

STRATEGY COMPONENTS:
1. VOLATILITY SPIKE (ATR(7)/ATR(30) > 2.0): Captures panic/extreme moves that revert
   - After vol spike, price typically mean-reverts within 2-5 bars
   - Works on ALL symbols (BTC, ETH, SOL) during crash/rally events

2. BOLLINGER BAND EXTREMES (20, 2.5): Entry trigger on vol spike
   - Long: price < lower BB + vol spike
   - Short: price > upper BB + vol spike
   - 2.5 std dev catches true extremes, not noise

3. 4h HMA(21) TREND BIAS: Via mtf_data helper (call ONCE before loop)
   - Only take LONG signals when price > 4h HMA (bullish bias)
   - Only take SHORT signals when price < 4h HMA (bearish bias)
   - HMA smoother than EMA, less lag

4. ADX HYSTERESIS FILTER: Avoid whipsaw in choppy markets
   - Enter only when ADX(14) > 25 (trending/volatile enough)
   - Exit when ADX(14) < 18 (market gone quiet)
   - Hysteresis gap (25→18) prevents rapid flip-flop

5. ATR TRAILING STOP (2.5x): Risk management
   - Signal → 0 when price moves 2.5*ATR against position
   - Protects from sustained moves against position

6. POSITION SIZING: 0.25 discrete (conservative for 30m volatility)
   - Max 25% capital per position
   - Discrete levels: 0.0, ±0.25 minimize fee churn

Why this should work:
- Vol spike + BB extreme = high-probability mean reversion (70%+ win rate)
- 4h HMA filter avoids catching falling knives in strong downtrends
- ADX hysteresis reduces trade count but improves quality
- Should generate 40-80 trades/year per symbol (enough for stats, not too many)
- Works on BTC, ETH, SOL individually (vol spikes happen on all)

Timeframe: 30m (REQUIRED for this experiment)
HTF: 4h via mtf_data helper (call ONCE before loop)
Position sizing: 0.25 discrete levels
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_vol_spike_bb_4h_hma_adx_hyst_atr_v1"
timeframe = "30m"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_bollinger_bands(close, period=20, std_dev=2.5):
    """Calculate Bollinger Bands with configurable std dev."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    return upper, lower, sma

def calculate_adx(high, low, close, period=14):
    """
    Calculate Average Directional Index (ADX).
    ADX > 25 = trending market
    ADX < 20 = ranging/choppy market
    """
    n = len(close)
    adx = np.full(n, np.nan)
    
    # Calculate True Range and Directional Movement
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    dm_plus = np.zeros(n)
    dm_minus = np.zeros(n)
    
    for i in range(1, n):
        if high[i] - high[i-1] > low[i-1] - low[i]:
            dm_plus[i] = max(0, high[i] - high[i-1])
        else:
            dm_plus[i] = 0
        
        if low[i-1] - low[i] > high[i] - high[i-1]:
            dm_minus[i] = max(0, low[i-1] - low[i])
        else:
            dm_minus[i] = 0
    
    # Smooth using Wilder's method (EMA with alpha=1/period)
    tr_smooth = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    dm_plus_smooth = pd.Series(dm_plus).ewm(span=period, min_periods=period, adjust=False).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # Calculate DI+ and DI-
    di_plus = np.zeros(n)
    di_minus = np.zeros(n)
    
    for i in range(period, n):
        if tr_smooth[i] > 1e-10:
            di_plus[i] = 100 * dm_plus_smooth[i] / tr_smooth[i]
            di_minus[i] = 100 * dm_minus_smooth[i] / tr_smooth[i]
    
    # Calculate DX and ADX
    dx = np.zeros(n)
    for i in range(period, n):
        di_sum = di_plus[i] + di_minus[i]
        if di_sum > 1e-10:
            dx[i] = 100 * np.abs(di_plus[i] - di_minus[i]) / di_sum
    
    # Smooth DX to get ADX
    adx_series = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean()
    adx[:] = adx_series.values
    
    return adx

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 30m indicators
    atr_14 = calculate_atr(high, low, close, 14)
    atr_7 = calculate_atr(high, low, close, 7)
    atr_30 = calculate_atr(high, low, close, 30)
    
    bb_upper, bb_lower, bb_sma = calculate_bollinger_bands(close, 20, 2.5)
    adx = calculate_adx(high, low, close, 14)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE = 0.25
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_close = 0.0
    lowest_close = 0.0
    entry_price = 0.0
    adx_entry = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(atr_7[i]) or np.isnan(atr_30[i]) or atr_30[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(adx[i]):
            signals[i] = 0.0
            continue
        
        # === VOLATILITY SPIKE DETECTION ===
        # ATR(7) / ATR(30) > 2.0 = volatility spike (panic/extreme move)
        vol_spike = (atr_7[i] / atr_30[i]) > 2.0
        
        # === BOLLINGER BAND EXTREMES ===
        price_below_lower_bb = close[i] < bb_lower[i]
        price_above_upper_bb = close[i] > bb_upper[i]
        
        # === 4h HMA TREND BIAS ===
        bull_trend_4h = close[i] > hma_4h_aligned[i]
        bear_trend_4h = close[i] < hma_4h_aligned[i]
        
        # === ADX HYSTERESIS FILTER ===
        # Enter when ADX > 25, exit when ADX < 18
        adx_high = adx[i] > 25
        adx_low = adx[i] < 18
        
        # === GENERATE SIGNAL ===
        new_signal = 0.0
        
        # VOL SPIKE + BB EXTREME + TREND BIAS = ENTRY
        if vol_spike:
            # LONG: vol spike + below lower BB + 4h bullish trend + ADX high enough
            if price_below_lower_bb and bull_trend_4h and adx_high:
                new_signal = SIZE
            
            # SHORT: vol spike + above upper BB + 4h bearish trend + ADX high enough
            elif price_above_upper_bb and bear_trend_4h and adx_high:
                new_signal = -SIZE
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest close for long position
                if close[i] > highest_close:
                    highest_close = close[i]
                stoploss_price = highest_close - 2.5 * atr_14[i]
                if close[i] < stoploss_price:
                    new_signal = 0.0
            
            if position_side < 0:
                # Update lowest close for short position
                if lowest_close == 0.0 or close[i] < lowest_close:
                    lowest_close = close[i]
                stoploss_price = lowest_close + 2.5 * atr_14[i]
                if close[i] > stoploss_price:
                    new_signal = 0.0
        
        # === ADX HYSTERESIS EXIT ===
        # Exit position if ADX drops below 18 (market gone quiet)
        if in_position and adx_low and new_signal == 0.0:
            new_signal = 0.0
        
        # === TREND REVERSAL EXIT ===
        # Exit long if 4h trend turns bearish
        if in_position and position_side > 0 and bear_trend_4h:
            new_signal = 0.0
        
        # Exit short if 4h trend turns bullish
        if in_position and position_side < 0 and bull_trend_4h:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
                adx_entry = adx[i]
            elif np.sign(new_signal) != position_side:
                # Position flip
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
                adx_entry = adx[i]
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_close = 0.0
                lowest_close = 0.0
                adx_entry = 0.0
        
        signals[i] = new_signal
    
    return signals