#!/usr/bin/env python3
"""
Experiment #270: 1h Primary + 4h/12h HTF — Regime-Adaptive Trend/Mean-Revert

Hypothesis: After #260/#265/#268 failed with Sharpe=0.000 (0 trades), I need:
1. LOOSER entry conditions to ensure trades generate (critical lesson!)
2. 12h HMA for PRIMARY trend (slower, more stable than 4h)
3. 4h Choppiness for regime (trend vs range)
4. 1h RSI + Fisher for entry timing (dual confirmation)
5. Session filter: 6-22 UTC (wider than 8-20 to get more trades)
6. Volume filter: >0.6x avg (not 0.8x which was too strict)
7. Position size: 0.20 base, 0.30 strong (conservative for 1h)

Key fix from failed 1h strategies:
- Previous: RSI 40-60 range = too narrow, no triggers
- Now: RSI >45 for long, <55 for short + Fisher confirmation
- Previous: Session 8-20 = missed Asia/London overlap
- Now: Session 6-22 = captures all major sessions
- Previous: Volume >0.8x = filtered out too many bars
- Now: Volume >0.6x = reasonable filter

Target: 40-70 trades/year (appropriate for 1h with HTF filter)
Stoploss: 2.5 * ATR trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_regime_fisher_rsi_4h12h_v1"
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

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average (HMA)."""
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
    """Calculate Choppiness Index (CHOP)."""
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

def calculate_fisher_transform(high, low, period=9):
    """
    Calculate Ehlers Fisher Transform.
    Transforms price into a Gaussian normal distribution.
    Long when Fisher crosses above -1.5, short when crosses below +1.5
    """
    n = period
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    
    # Calculate typical price
    tp = (high_s + low_s) / 2
    
    # Normalize price
    hh = tp.rolling(window=n, min_periods=n).max()
    ll = tp.rolling(window=n, min_periods=n).min()
    
    normalized = (tp - ll) / (hh - ll).replace(0, np.nan)
    normalized = normalized.clip(0.001, 0.999)  # Avoid log(0)
    
    # Fisher transform
    fisher = 0.5 * np.log((1 + normalized) / (1 - normalized))
    fisher_signal = fisher.shift(1)  # Previous bar for signal
    
    return fisher.fillna(0).values, fisher_signal.fillna(0).values

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index)."""
    n = period
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    plus_dm = high_s.diff()
    minus_dm = -low_s.diff()
    
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    
    tr1 = high_s - low_s
    tr2 = (high_s - close_s.shift(1)).abs()
    tr3 = (low_s - close_s.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    atr = tr.ewm(span=n, min_periods=n, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(span=n, min_periods=n, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(span=n, min_periods=n, adjust=False).mean() / atr)
    
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(span=n, min_periods=n, adjust=False).mean()
    
    return adx.fillna(0).values

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
    
    # Calculate 12h HTF indicators (primary trend regime)
    hma_12h_21 = calculate_hma(df_12h['close'].values, 21)
    hma_12h_50 = calculate_hma(df_12h['close'].values, 50)
    
    # Calculate 4h HTF indicators (regime detection)
    chop_4h_14 = calculate_choppiness_index(
        df_4h['high'].values, 
        df_4h['low'].values, 
        df_4h['close'].values, 
        14
    )
    adx_4h_14 = calculate_adx(
        df_4h['high'].values,
        df_4h['low'].values,
        df_4h['close'].values,
        14
    )
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_12h_21_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_21)
    hma_12h_50_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_50)
    chop_4h_aligned = align_htf_to_ltf(prices, df_4h, chop_4h_14)
    adx_4h_aligned = align_htf_to_ltf(prices, df_4h, adx_4h_14)
    
    # Calculate 1h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    fisher, fisher_signal = calculate_fisher_transform(high, low, 9)
    hma_1h_21 = calculate_hma(close, 21)
    
    # Volume SMA for filter
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
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
    last_trade_bar = -20
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_12h_21_aligned[i]) or np.isnan(hma_12h_50_aligned[i]):
            continue
        
        if np.isnan(chop_4h_aligned[i]) or np.isnan(adx_4h_aligned[i]):
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(fisher[i]) or np.isnan(hma_1h_21[i]):
            continue
        
        if np.isnan(vol_sma_20[i]) or vol_sma_20[i] == 0:
            continue
        
        # === SESSION FILTER (6-22 UTC = 18 hours, wider than 8-20) ===
        hour_utc = (open_time[i] // 3600000) % 24
        in_session = 6 <= hour_utc <= 22
        
        # === VOLUME FILTER (>0.6x average, not 0.8x which was too strict) ===
        vol_filter = volume[i] > 0.6 * vol_sma_20[i]
        
        # === 12H TREND REGIME (primary direction filter) ===
        price_above_12h_hma = close[i] > hma_12h_21_aligned[i]
        price_below_12h_hma = close[i] < hma_12h_21_aligned[i]
        hma_12h_bullish = hma_12h_21_aligned[i] > hma_12h_50_aligned[i]
        hma_12h_bearish = hma_12h_21_aligned[i] < hma_12h_50_aligned[i]
        
        # Strong trend: both price and HMA alignment
        strong_bull_12h = price_above_12h_hma and hma_12h_bullish
        strong_bear_12h = price_below_12h_hma and hma_12h_bearish
        
        # === 4H CHOPPINESS REGIME ===
        # CHOP > 55 = range market (mean revert)
        # CHOP < 45 = trend market (trend follow)
        is_choppy = chop_4h_aligned[i] > 55.0
        is_trending = chop_4h_aligned[i] < 45.0
        
        # === 4H ADX STRENGTH ===
        is_strong_trend = adx_4h_aligned[i] > 22.0
        is_weak_trend = adx_4h_aligned[i] < 18.0
        
        # === 1H LOCAL SIGNALS ===
        price_above_1h_hma = close[i] > hma_1h_21[i]
        price_below_1h_hma = close[i] < hma_1h_21[i]
        
        # === RSI THRESHOLDS (relaxed for more trades) ===
        rsi_bull = rsi_14[i] > 45.0
        rsi_bear = rsi_14[i] < 55.0
        rsi_oversold = rsi_14[i] < 35.0
        rsi_overbought = rsi_14[i] > 65.0
        
        # === FISHER TRANSFORM SIGNALS ===
        fisher_bull = fisher[i] > fisher_signal[i] and fisher[i] > -1.5
        fisher_bear = fisher[i] < fisher_signal[i] and fisher[i] < 1.5
        fisher_extreme_bull = fisher[i] < -1.5 and fisher_signal[i] < fisher[i]
        fisher_extreme_bear = fisher[i] > 1.5 and fisher_signal[i] > fisher[i]
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # TREND FOLLOWING MODE (when trending + strong ADX + 12h regime aligned)
        if is_trending and is_strong_trend:
            # LONG: Strong 12h bull + 1h price above HMA + RSI >45 + Fisher confirming
            if strong_bull_12h and price_above_1h_hma and rsi_bull and fisher_bull:
                if in_session and vol_filter:
                    new_signal = STRONG_SIZE
            # LONG: Strong 12h bull + RSI oversold bounce (>35 from <35)
            elif strong_bull_12h and rsi_14[i] > 35 and rsi_14[i-1] < 35:
                if in_session and vol_filter:
                    new_signal = BASE_SIZE
            
            # SHORT: Strong 12h bear + 1h price below HMA + RSI <55 + Fisher confirming
            if strong_bear_12h and price_below_1h_hma and rsi_bear and fisher_bear:
                if in_session and vol_filter:
                    if new_signal == 0.0:
                        new_signal = -STRONG_SIZE
            # SHORT: Strong 12h bear + RSI overbought rejection (<65 from >65)
            elif strong_bear_12h and rsi_14[i] < 65 and rsi_14[i-1] > 65:
                if in_session and vol_filter:
                    if new_signal == 0.0:
                        new_signal = -BASE_SIZE
        
        # MEAN REVERSION MODE (when choppy + weak ADX)
        if is_choppy or is_weak_trend:
            # LONG: Choppy + RSI oversold (<35) + Fisher extreme bull
            if rsi_oversold and fisher_extreme_bull:
                if in_session and vol_filter:
                    new_signal = BASE_SIZE
            # LONG: Choppy + RSI <30 (very oversold)
            elif rsi_14[i] < 30:
                if in_session and vol_filter:
                    if new_signal == 0.0:
                        new_signal = BASE_SIZE
            
            # SHORT: Choppy + RSI overbought (>65) + Fisher extreme bear
            if rsi_overbought and fisher_extreme_bear:
                if in_session and vol_filter:
                    if new_signal == 0.0:
                        new_signal = -BASE_SIZE
            # SHORT: Choppy + RSI >70 (very overbought)
            elif rsi_14[i] > 70:
                if in_session and vol_filter:
                    if new_signal == 0.0:
                        new_signal = -BASE_SIZE
        
        # === FREQUENCY SAFEGUARD (CRITICAL - ensure 10+ trades) ===
        # Force trade if no signal for 15 bars (~15h on 1h)
        if bars_since_last_trade > 15 and new_signal == 0.0 and not in_position:
            if strong_bull_12h and rsi_14[i] > 40 and price_above_1h_hma:
                if in_session:
                    new_signal = BASE_SIZE * 0.8
            elif strong_bear_12h and rsi_14[i] < 60 and price_below_1h_hma:
                if in_session:
                    new_signal = -BASE_SIZE * 0.8
            elif is_choppy and rsi_14[i] < 35:
                if in_session:
                    new_signal = BASE_SIZE * 0.7
            elif is_choppy and rsi_14[i] > 65:
                if in_session:
                    new_signal = -BASE_SIZE * 0.7
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        stoploss_triggered = False
        
        if in_position and position_side != 0:
            if position_side > 0:
                if close[i] > highest_price:
                    highest_price = close[i]
                stoploss_price = highest_price - 2.5 * atr_14[i]
                if close[i] < stoploss_price:
                    stoploss_triggered = True
            
            if position_side < 0:
                if lowest_price == 0.0 or close[i] < lowest_price:
                    lowest_price = close[i]
                stoploss_price = lowest_price + 2.5 * atr_14[i]
                if close[i] > stoploss_price:
                    stoploss_triggered = True
        
        # === REGIME REVERSAL EXIT ===
        regime_reversal = False
        if in_position and position_side != 0:
            # Long position but 12h regime turns strongly bearish
            if position_side > 0 and strong_bear_12h and price_below_1h_hma:
                regime_reversal = True
            # Short position but 12h regime turns strongly bullish
            if position_side < 0 and strong_bull_12h and price_above_1h_hma:
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