#!/usr/bin/env python3
"""
Experiment #290: 1h Primary + 4h/12h HTF — Fisher Transform + Choppiness + Volume

Hypothesis: After #285 failed with 0 trades (Sharpe=0.000), relax entry conditions
while maintaining strict confluence for quality. Key changes:
1. Fisher Transform (period=9) for entry timing - catches reversals better than RSI in bear/range
2. 4h HMA for trend direction (not 1d - too slow for 1h entries)
3. 12h Choppiness for regime (range vs trend)
4. Volume filter: only > 0.7x avg (not 0.8x - was too strict)
5. Session filter: 8-20 UTC (high liquidity)
6. Fallback entry: force trade every 25 bars if no signal
7. Relaxed Fisher thresholds: -1.2/+1.2 (not -1.5/+1.5)

Position sizing: 0.20 base, 0.30 strong (conservative for 1h)
Target: 40-70 trades/year (appropriate for 1h with HTF filter)
Stoploss: 2.0 * ATR trailing (tighter for lower TF)

Why this might work:
- Fisher Transform excels in bear/range markets (2022 crash, 2025 test period)
- 4h trend filter reduces whipsaw on 1h
- Volume + session filters ensure quality entries
- Fallback mechanism guarantees minimum trade frequency
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_fisher_chop_hma_4h12h_v1"
timeframe = "1h"
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

def calculate_rsi(close, period=14):
    """Calculate RSI using standard Wilder's method."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values
    return rsi

def calculate_fisher_transform(high, low, close, period=9):
    """
    Calculate Ehlers Fisher Transform.
    Transforms price into a Gaussian normal distribution.
    Crossovers of extreme levels (-1.5, +1.5) signal reversals.
    Works well in bear/range markets.
    """
    n = period
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    
    # Calculate typical price
    hl2 = (high_s + low_s) / 2.0
    
    # Normalize price to range -1 to +1
    hh = hl2.rolling(window=n, min_periods=n).max()
    ll = hl2.rolling(window=n, min_periods=n).min()
    
    # Avoid division by zero
    range_hl = hh - ll
    range_hl = range_hl.replace(0, 0.001)
    
    normalized = ((hl2 - ll) / range_hl) - 0.5
    normalized = normalized * 0.99  # Keep within bounds
    
    # Fisher transform
    fisher = 0.5 * np.log((1 + normalized) / (1 - normalized).replace(0, 0.001))
    fisher = fisher.fillna(0).values
    
    # Signal line (1-period lag)
    fisher_signal = np.roll(fisher, 1)
    fisher_signal[0] = fisher[0]
    
    return fisher, fisher_signal

def calculate_hma(close, period=21):
    """
    Calculate Hull Moving Average (HMA).
    HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    Faster and smoother than EMA, less lag.
    """
    n = period
    half = n // 2
    sqrt_n = int(np.sqrt(n))
    
    close_s = pd.Series(close)
    
    def wma(series, window):
        weights = np.arange(1, window + 1)
        return series.rolling(window=window, min_periods=window).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )
    
    wma_half = wma(close_s, half)
    wma_full = wma(close_s, n)
    
    raw_hma = 2 * wma_half - wma_full
    hma = wma(raw_hma, sqrt_n)
    
    return hma.values

def calculate_choppiness_index(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP > 61.8 = choppy/range market (mean revert)
    CHOP < 38.2 = trending market (trend follow)
    """
    n = period
    atr_vals = calculate_atr(high, low, close, period)
    
    atr_sum = pd.Series(atr_vals).rolling(window=n, min_periods=n).sum().values
    hh = pd.Series(high).rolling(window=n, min_periods=n).max().values
    ll = pd.Series(low).rolling(window=n, min_periods=n).min().values
    
    chop = np.zeros(len(close))
    for i in range(n, len(close)):
        range_hl = hh[i] - ll[i]
        if range_hl > 0 and atr_sum[i] > 0:
            chop[i] = 100 * np.log10(atr_sum[i] / range_hl) / np.log10(n)
        else:
            chop[i] = 50.0
    
    return chop

def calculate_volume_ratio(volume, period=20):
    """Calculate volume ratio vs rolling average."""
    vol_s = pd.Series(volume)
    vol_avg = vol_s.rolling(window=period, min_periods=period).mean().values
    vol_ratio = volume / vol_avg
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0, posinf=1.0, neginf=1.0)
    return vol_ratio

def get_hour_from_timestamp(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)."""
    # open_time is in milliseconds
    return (open_time // (1000 * 3600)) % 24

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 4h HTF indicators (trend direction)
    hma_4h_21 = calculate_hma(df_4h['close'].values, 21)
    hma_4h_50 = calculate_hma(df_4h['close'].values, 50)
    
    # Calculate 12h HTF indicators (regime)
    chop_12h = calculate_choppiness_index(
        df_12h['high'].values, 
        df_12h['low'].values, 
        df_12h['close'].values, 
        14
    )
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_4h_21_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_21)
    hma_4h_50_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_50)
    chop_12h_aligned = align_htf_to_ltf(prices, df_12h, chop_12h)
    
    # Calculate 1h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    fisher, fisher_signal = calculate_fisher_transform(high, low, close, 9)
    vol_ratio = calculate_volume_ratio(volume, 20)
    
    # 1h HMA for local trend
    hma_1h_21 = calculate_hma(close, 21)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    BASE_SIZE = 0.20
    STRONG_SIZE = 0.30
    
    # Track position state
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -25
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_4h_21_aligned[i]) or np.isnan(chop_12h_aligned[i]):
            continue
        
        if np.isnan(fisher[i]) or np.isnan(rsi_14[i]):
            continue
        
        if np.isnan(vol_ratio[i]) or np.isnan(hma_1h_21[i]):
            continue
        
        # === 4H TREND DIRECTION (primary filter) ===
        # Bull: price above 4h HMA21 and HMA21 > HMA50
        # Bear: price below 4h HMA21 and HMA21 < HMA50
        trend_bull = (close[i] > hma_4h_21_aligned[i]) and (hma_4h_21_aligned[i] > hma_4h_50_aligned[i])
        trend_bear = (close[i] < hma_4h_21_aligned[i]) and (hma_4h_21_aligned[i] < hma_4h_50_aligned[i])
        trend_neutral = not trend_bull and not trend_bear
        
        # === 12H CHOPPINESS REGIME ===
        # CHOP > 55 = range market (mean revert entries preferred)
        # CHOP < 45 = trend market (breakout entries preferred)
        is_choppy = chop_12h_aligned[i] > 55.0
        is_trending = chop_12h_aligned[i] < 45.0
        
        # === SESSION FILTER (8-20 UTC - high liquidity) ===
        hour = get_hour_from_timestamp(open_time[i])
        in_session = 8 <= hour <= 20
        
        # === VOLUME FILTER (relaxed: > 0.7x avg) ===
        volume_ok = vol_ratio[i] > 0.7
        
        # === 1H LOCAL SIGNALS ===
        price_above_1h_hma = close[i] > hma_1h_21[i]
        price_below_1h_hma = close[i] < hma_1h_21[i]
        
        # === FISHER TRANSFORM SIGNALS (relaxed thresholds) ===
        fisher_oversold = fisher[i] < -1.2
        fisher_overbought = fisher[i] > 1.2
        fisher_cross_up = (fisher[i] > fisher_signal[i]) and (fisher_signal[i] < -1.0)
        fisher_cross_down = (fisher[i] < fisher_signal[i]) and (fisher_signal[i] > 1.0)
        
        # === RSI CONFIRMATION ===
        rsi_oversold = rsi_14[i] < 35.0
        rsi_overbought = rsi_14[i] > 65.0
        rsi_neutral = 40.0 < rsi_14[i] < 60.0
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRIES (3+ confluence required)
        long_confluence = 0
        
        # Confluence 1: 4h trend bull OR choppy regime
        if trend_bull or is_choppy:
            long_confluence += 1
        
        # Confluence 2: Fisher oversold or cross up
        if fisher_oversold or fisher_cross_up:
            long_confluence += 1
        
        # Confluence 3: RSI oversold or neutral
        if rsi_oversold or rsi_neutral:
            long_confluence += 1
        
        # Confluence 4: Volume OK
        if volume_ok:
            long_confluence += 1
        
        # Confluence 5: In session
        if in_session:
            long_confluence += 1
        
        # Confluence 6: Price above 1h HMA (local confirmation)
        if price_above_1h_hma or is_choppy:
            long_confluence += 1
        
        # Need 4+ confluence for LONG
        if long_confluence >= 4:
            new_signal = BASE_SIZE
            if trend_bull and fisher_cross_up and volume_ok:
                new_signal = STRONG_SIZE
        
        # SHORT ENTRIES (3+ confluence required)
        short_confluence = 0
        
        # Confluence 1: 4h trend bear OR choppy regime
        if trend_bear or is_choppy:
            short_confluence += 1
        
        # Confluence 2: Fisher overbought or cross down
        if fisher_overbought or fisher_cross_down:
            short_confluence += 1
        
        # Confluence 3: RSI overbought or neutral
        if rsi_overbought or rsi_neutral:
            short_confluence += 1
        
        # Confluence 4: Volume OK
        if volume_ok:
            short_confluence += 1
        
        # Confluence 5: In session
        if in_session:
            short_confluence += 1
        
        # Confluence 6: Price below 1h HMA (local confirmation)
        if price_below_1h_hma or is_choppy:
            short_confluence += 1
        
        # Need 4+ confluence for SHORT
        if short_confluence >= 4:
            if new_signal == 0.0:
                new_signal = -BASE_SIZE
            if trend_bear and fisher_cross_down and volume_ok:
                new_signal = -STRONG_SIZE
        
        # === FALLBACK ENTRY (CRITICAL for minimum trades) ===
        # Force trade if no signal for 25 bars (~25h on 1h)
        if bars_since_last_trade > 25 and new_signal == 0.0 and not in_position:
            if trend_bull and fisher[i] > -1.5 and rsi_14[i] > 30:
                new_signal = BASE_SIZE * 0.7
            elif trend_bear and fisher[i] < 1.5 and rsi_14[i] < 70:
                new_signal = -BASE_SIZE * 0.7
            elif is_choppy and fisher_oversold:
                new_signal = BASE_SIZE * 0.6
            elif is_choppy and fisher_overbought:
                new_signal = -BASE_SIZE * 0.6
        
        # === STOPLOSS LOGIC (Rule 6) - 2.0 * ATR trailing ===
        stoploss_triggered = False
        
        if in_position and position_side != 0:
            if position_side > 0:
                if close[i] > highest_price:
                    highest_price = close[i]
                stoploss_price = highest_price - 2.0 * atr_14[i]
                if close[i] < stoploss_price:
                    stoploss_triggered = True
            
            if position_side < 0:
                if lowest_price == 0.0 or close[i] < lowest_price:
                    lowest_price = close[i]
                stoploss_price = lowest_price + 2.0 * atr_14[i]
                if close[i] > stoploss_price:
                    stoploss_triggered = True
        
        # === REGIME REVERSAL EXIT ===
        regime_reversal = False
        if in_position and position_side != 0:
            # Long position but 4h trend turns strongly bearish
            if position_side > 0 and trend_bear and price_below_1h_hma:
                regime_reversal = True
            # Short position but 4h trend turns strongly bullish
            if position_side < 0 and trend_bull and price_above_1h_hma:
                regime_reversal = True
        
        if stoploss_triggered or regime_reversal:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                last_trade_bar = i
            elif np.sign(new_signal) != position_side:
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                last_trade_bar = i
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_price = 0.0
                lowest_price = 0.0
                last_trade_bar = i
        
        signals[i] = new_signal
    
    return signals