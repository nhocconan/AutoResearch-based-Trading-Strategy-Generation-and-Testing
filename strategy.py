#!/usr/bin/env python3
"""
Experiment #1601: 15m Primary + 1h/4h/1d HTF — CPR Pivot + RSI Pullback + Session Filter

Hypothesis: 15m strategies fail due to overtrading. This uses DAILY CPR (Central Pivot Range)
from 1d HTF as key S/R levels, 4h HMA for trend bias, and 15m RSI(7) for precise pullback
entries. Session filter (00-12 UTC) reduces trades to 40-100/year while capturing London/NY
overlap liquidity.

Key innovations:
1. DAILY CPR LEVELS: BC/TC/Pivot from 1d HTF — institutional S/R that crypto respects
2. REGIME FILTER: Choppiness Index distinguishes trend (CHOP<38) vs range (CHOP>61)
3. PULLBACK ENTRY: Long in uptrend when RSI(7)<40 near TC support, short when RSI(7)>60 near BC
4. SESSION FILTER: Only trade 00-12 UTC (reduces 60% of trades, keeps high-liquidity hours)
5. VOLUME CONFIRMATION: Require 1.2x avg volume for breakout entries
6. DISCRETE SIZING: 0.15-0.20 (smaller for 15m frequency), stoploss at 2.5x ATR

Why this should beat failed 15m attempts:
- Previous 15m strategies had NO session filter → 300+ trades/year → fee drag
- Previous 15m strategies had NO HTF bias → counter-trend trades in strong trends
- CPR levels are self-calculating from OHLC, no external data needed
- RSI(7) is faster than RSI(14), catches pullbacks earlier

Entry logic:
- LONG trend: 4h_HMA bullish + CHOP<38 + price>TC + RSI(7)<45 + session 00-12 UTC
- SHORT trend: 4h_HMA bearish + CHOP<38 + price<BC + RSI(7)>55 + session 00-12 UTC
- LONG range: CHOP>61 + price<BC + RSI(7)<35 (fade lower bound)
- SHORT range: CHOP>61 + price>TC + RSI(7)>65 (fade upper bound)

Target: Sharpe>0.6, trades>=30 train, trades>=3 test, DD>-35%, trades/year<100
Timeframe: 15m
Size: 0.15-0.20 discrete (smaller than 1h/4h due to higher frequency)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_cpr_rsi_session_4h1d_v1"
timeframe = "15m"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average - reduces lag while smoothing"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = max(1, period // 2)
    sqrt_n = max(1, int(np.sqrt(period)))
    
    def wma(series, span):
        result = np.full(len(series), np.nan, dtype=np.float64)
        weights = np.arange(1, span + 1, dtype=np.float64)
        weight_sum = np.sum(weights)
        for i in range(span - 1, len(series)):
            if not np.isnan(series[i]):
                window = series[i - span + 1:i + 1].astype(np.float64)
                if not np.any(np.isnan(window)):
                    result[i] = np.sum(window * weights) / weight_sum
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    diff = np.full(n, np.nan, dtype=np.float64)
    for i in range(period - 1, n):
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            diff[i] = 2.0 * wma_half[i] - wma_full[i]
    
    return wma(diff, sqrt_n)

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
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
    """Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    gain = np.insert(gain, 0, 0)
    loss = np.insert(loss, 0, 0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.full(n, np.nan, dtype=np.float64)
    mask = avg_loss != 0
    rs = np.zeros(n)
    rs[mask] = avg_gain[mask] / avg_loss[mask]
    rsi[mask] = 100 - (100 / (1 + rs[mask]))
    
    return rsi

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index - measures market choppy vs trending
    CHOP > 61.8 = ranging, CHOP < 38.2 = trending
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    chop = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period, n):
        atr_sum = np.sum(tr[i - period + 1:i + 1])
        highest_high = np.max(high[i - period + 1:i + 1])
        lowest_low = np.min(low[i - period + 1:i + 1])
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10 and atr_sum > 0:
            chop[i] = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
    
    return chop

def calculate_cpr_levels(open_prev, high_prev, low_prev, close_prev):
    """
    Central Pivot Range (CPR) - daily support/resistance levels
    BC = Bottom Central, TC = Top Central, Pivot = standard pivot
    
    Formula:
    Pivot = (High + Low + Close) / 3
    BC = (High + Low) / 2
    TC = (Pivot + BC) / 2
    """
    n = len(open_prev)
    
    pivot = np.full(n, np.nan, dtype=np.float64)
    bc = np.full(n, np.nan, dtype=np.float64)
    tc = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(n):
        if not np.isnan(high_prev[i]) and not np.isnan(low_prev[i]) and not np.isnan(close_prev[i]):
            pivot[i] = (high_prev[i] + low_prev[i] + close_prev[i]) / 3.0
            bc[i] = (high_prev[i] + low_prev[i]) / 2.0
            tc[i] = (pivot[i] + bc[i]) / 2.0
    
    return pivot, bc, tc

def calculate_volume_ratio(volume, period=20):
    """Current volume vs average volume"""
    n = len(volume)
    if n < period:
        return np.full(n, np.nan)
    
    vol_avg = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    vol_ratio = volume / vol_avg
    
    return vol_ratio

def get_utc_hour(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)"""
    # open_time is in milliseconds since epoch
    return (open_time // (1000 * 60 * 60)) % 24

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1h = get_htf_data(prices, '1h')
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF indicators
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate CPR levels from 1d data (need previous day's OHLC)
    open_1d = df_1d['open'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Shift by 1 to get PREVIOUS day's CPR (no look-ahead)
    pivot_1d_raw, bc_1d_raw, tc_1d_raw = calculate_cpr_levels(
        np.roll(open_1d, 1), np.roll(high_1d, 1), np.roll(low_1d, 1), np.roll(close_1d, 1)
    )
    pivot_1d_raw[0] = np.nan
    bc_1d_raw[0] = np.nan
    tc_1d_raw[0] = np.nan
    
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d_raw)
    bc_1d_aligned = align_htf_to_ltf(prices, df_1d, bc_1d_raw)
    tc_1d_aligned = align_htf_to_ltf(prices, df_1d, tc_1d_raw)
    
    # Calculate 15m indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_7 = calculate_rsi(close, period=7)
    rsi_14 = calculate_rsi(close, period=14)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    vol_ratio = calculate_volume_ratio(volume, period=20)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.15
    SIZE_STRONG = 0.20
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Warmup period
    min_bars = 100
    
    for i in range(min_bars, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(rsi_7[i]) or np.isnan(chop_14[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_4h_aligned[i]) or np.isnan(pivot_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === SESSION FILTER (00-12 UTC only) ===
        utc_hour = get_utc_hour(open_time[i])
        in_session = 0 <= utc_hour < 12
        
        # === REGIME DETECTION (Choppiness Index) ===
        chop = chop_14[i]
        is_trend_regime = chop < 38.2
        is_range_regime = chop > 61.8
        
        # === TREND DIRECTION (4h HMA bias) ===
        price_above_4h = close[i] > hma_4h_aligned[i]
        price_below_4h = close[i] < hma_4h_aligned[i]
        
        # === CPR LEVELS ===
        pivot = pivot_1d_aligned[i]
        bc = bc_1d_aligned[i]
        tc = tc_1d_aligned[i]
        
        # Price position relative to CPR
        price_above_tc = close[i] > tc
        price_below_bc = close[i] < bc
        price_in_cpr = (close[i] >= bc) and (close[i] <= tc)
        
        # Narrow CPR = potential breakout day
        cpr_width = (tc - bc) / bc if bc > 0 else 1.0
        narrow_cpr = cpr_width < 0.005  # CPR width < 0.5%
        
        # === RSI SIGNALS ===
        rsi_oversold = rsi_7[i] < 40
        rsi_overbought = rsi_7[i] > 60
        rsi_extreme_low = rsi_7[i] < 30
        rsi_extreme_high = rsi_7[i] > 70
        
        # === VOLUME CONFIRMATION ===
        vol_confirmed = vol_ratio[i] > 1.2 if not np.isnan(vol_ratio[i]) else False
        
        # === ENTRY LOGIC (SELECTIVE - session + confluence) ===
        desired_signal = 0.0
        
        # Only trade during session hours
        if not in_session:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # TREND REGIME: Follow 4h trend, enter on RSI pullback near CPR
        if is_trend_regime:
            # LONG: 4h bullish + price above TC + RSI pullback < 45
            if price_above_4h and price_above_tc and rsi_oversold:
                desired_signal = SIZE_STRONG if vol_confirmed else SIZE_BASE
            
            # SHORT: 4h bearish + price below BC + RSI pullback > 55
            elif price_below_4h and price_below_bc and rsi_overbought:
                desired_signal = -SIZE_STRONG if vol_confirmed else -SIZE_BASE
        
        # RANGE REGIME: Fade CPR extremes with RSI extremes
        elif is_range_regime:
            # LONG: Price at/near BC + RSI extreme low
            if price_below_bc and rsi_extreme_low:
                desired_signal = SIZE_BASE
            
            # SHORT: Price at/near TC + RSI extreme high
            elif price_above_tc and rsi_extreme_high:
                desired_signal = -SIZE_BASE
        
        # NEUTRAL REGIME: Use CPR breakout with narrow CPR
        else:
            # Narrow CPR breakout = strong directional move likely
            if narrow_cpr:
                # LONG: Break above TC + 4h bullish + RSI not overbought
                if price_above_tc and price_above_4h and rsi_7[i] < 65:
                    desired_signal = SIZE_BASE
                
                # SHORT: Break below BC + 4h bearish + RSI not oversold
                elif price_below_bc and price_below_4h and rsi_7[i] > 35:
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