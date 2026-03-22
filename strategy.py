#!/usr/bin/env python3
"""
Experiment #253: 1d Primary + 1w HTF — HMA Trend + ADX Regime + RSI Entries

Hypothesis: After analyzing 250+ experiments, 1d strategies fail when:
1. Entry conditions are too strict (need 5+ conditions to align)
2. Using daily indicators for regime (too noisy) — weekly is better
3. Choppiness Index on daily is unreliable (failed #243, #247, #252)

NEW APPROACH for 1d:
1. 1w HMA(21) slope for PRIMARY trend regime (stable, weekly filter)
2. ADX(14) on 1d for trend strength (ADX>25=trend, ADX<20=range)
3. RSI(14) on 1d with WIDE thresholds (25/75 for mean revert, 40/60 for trend)
4. Volume confirmation (vol > 20d SMA) to filter fake breakouts
5. ATR(14) for 3x trailing stops (wider on daily timeframe)
6. Force trade every 30 bars (~30 days) to ensure 10+ trades/year

Key differences from failed 1d strategies:
- WEEKLY HMA for regime (not daily) — more stable trend filter
- ADX instead of Choppiness — proven on 12h (#246 worked)
- Fewer entry conditions (max 3 AND conditions)
- Force trade every 30 bars (not 20, daily is slower)
- Volume filter to avoid low-liquidity traps

Position sizing: 0.25 base, 0.35 strong (discrete levels)
Target: 20-40 trades/year per symbol (within 1d cost model)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_hma_adx_rsi_regime_1w_v1"
timeframe = "1d"
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
    """
    Calculate Hull Moving Average (HMA).
    HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n)
    """
    close_s = pd.Series(close)
    half = int(period / 2)
    sqrt_n = int(np.sqrt(period))
    
    wma_half = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma_full = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    
    raw_hma = 2 * wma_half - wma_full
    hma = raw_hma.ewm(span=sqrt_n, min_periods=sqrt_n, adjust=False).mean()
    
    return hma.values

def calculate_adx(high, low, close, period=14):
    """
    Calculate ADX (Average Directional Index).
    ADX > 25 = trending market
    ADX < 20 = ranging/choppy market
    """
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
    tr = tr1.combine(tr2, np.maximum).combine(tr3, np.maximum)
    
    atr = tr.ewm(span=period, min_periods=period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(span=period, min_periods=period, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(span=period, min_periods=period, adjust=False).mean() / atr)
    
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(span=period, min_periods=period, adjust=False).mean()
    
    return adx.fillna(0).values

def calculate_hma_slope(hma_values, lookback=3):
    """Calculate HMA slope as percentage change over lookback."""
    slope = np.zeros(len(hma_values))
    for i in range(lookback, len(hma_values)):
        prev = hma_values[i - lookback]
        curr = hma_values[i]
        if prev != 0 and not np.isnan(prev) and not np.isnan(curr):
            slope[i] = (curr - prev) / abs(prev) * 100
    return slope

def calculate_volume_sma(volume, period=20):
    """Calculate SMA of volume."""
    vol_sma = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    return vol_sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w HTF indicators (primary trend regime)
    hma_1w_21 = calculate_hma(df_1w['close'].values, 21)
    hma_1w_slope = calculate_hma_slope(hma_1w_21, 2)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_1w_21_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_21)
    hma_1w_slope_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_slope)
    
    # Calculate 1d indicators
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    adx_14 = calculate_adx(high, low, close, 14)
    hma_1d_21 = calculate_hma(close, 21)
    hma_1d_50 = calculate_hma(close, 50)
    vol_sma_20 = calculate_volume_sma(volume, 20)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    BASE_SIZE = 0.25
    STRONG_SIZE = 0.35
    
    # Track position state
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -30
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_1w_21_aligned[i]) or np.isnan(hma_1w_slope_aligned[i]):
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(adx_14[i]):
            continue
        
        if np.isnan(hma_1d_21[i]) or np.isnan(vol_sma_20[i]):
            continue
        
        # === 1W TREND REGIME (primary direction filter) ===
        # Bull: 1w HMA slope > 0.10%
        # Bear: 1w HMA slope < -0.10%
        regime_bull = hma_1w_slope_aligned[i] > 0.10
        regime_bear = hma_1w_slope_aligned[i] < -0.10
        regime_neutral = not regime_bull and not regime_bear
        
        price_above_1w_hma = close[i] > hma_1w_21_aligned[i]
        price_below_1w_hma = close[i] < hma_1w_21_aligned[i]
        
        # === ADX REGIME ===
        # ADX > 25 = trending market (trend follow)
        # ADX < 20 = ranging market (mean revert)
        is_trending = adx_14[i] > 25.0
        is_ranging = adx_14[i] < 20.0
        
        # === VOLUME CONFIRMATION ===
        vol_confirm = volume[i] > vol_sma_20[i]
        
        # === 1D LOCAL SIGNALS ===
        price_above_1d_hma = close[i] > hma_1d_21[i]
        price_below_1d_hma = close[i] < hma_1d_21[i]
        hma_1d_bullish = hma_1d_21[i] > hma_1d_50[i]
        hma_1d_bearish = hma_1d_21[i] < hma_1d_50[i]
        
        # === RSI THRESHOLDS ===
        rsi_oversold = rsi_14[i] < 25.0
        rsi_overbought = rsi_14[i] > 75.0
        rsi_bull_confirm = rsi_14[i] > 40.0
        rsi_bear_confirm = rsi_14[i] < 60.0
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # TREND FOLLOWING MODE (when ADX > 25)
        if is_trending:
            # LONG: Trending + bull regime + price above 1d HMA + RSI confirming + volume
            if regime_bull and price_above_1d_hma and rsi_bull_confirm and vol_confirm:
                new_signal = STRONG_SIZE
            # LONG: Trending + price above 1w HMA + 1d HMA bullish
            elif price_above_1w_hma and hma_1d_bullish and rsi_14[i] > 35:
                new_signal = BASE_SIZE
            
            # SHORT: Trending + bear regime + price below 1d HMA + RSI confirming + volume
            if regime_bear and price_below_1d_hma and rsi_bear_confirm and vol_confirm:
                if new_signal == 0.0:
                    new_signal = -STRONG_SIZE
            # SHORT: Trending + price below 1w HMA + 1d HMA bearish
            elif price_below_1w_hma and hma_1d_bearish and rsi_14[i] < 65:
                if new_signal == 0.0:
                    new_signal = -BASE_SIZE
        
        # MEAN REVERSION MODE (when ADX < 20)
        if is_ranging:
            # LONG: Ranging + RSI oversold (<25) + not in strong bear regime
            if rsi_oversold and not regime_bear:
                new_signal = BASE_SIZE
            # LONG: Ranging + RSI very oversold (<20) in any regime
            if rsi_14[i] < 20:
                if new_signal == 0.0:
                    new_signal = BASE_SIZE * 0.8
            
            # SHORT: Ranging + RSI overbought (>75) + not in strong bull regime
            if rsi_overbought and not regime_bull:
                if new_signal == 0.0:
                    new_signal = -BASE_SIZE
            # SHORT: Ranging + RSI very overbought (>80) in any regime
            if rsi_14[i] > 80:
                if new_signal == 0.0:
                    new_signal = -BASE_SIZE * 0.8
        
        # === FREQUENCY SAFEGUARD (CRITICAL for 10+ trades) ===
        # Force trade if no signal for 30 bars (~30 days on 1d)
        if bars_since_last_trade > 30 and new_signal == 0.0 and not in_position:
            if regime_bull and rsi_14[i] > 35 and price_above_1d_hma:
                new_signal = BASE_SIZE * 0.6
            elif regime_bear and rsi_14[i] < 65 and price_below_1d_hma:
                new_signal = -BASE_SIZE * 0.6
            elif is_ranging and rsi_14[i] < 30:
                new_signal = BASE_SIZE * 0.5
            elif is_ranging and rsi_14[i] > 70:
                new_signal = -BASE_SIZE * 0.5
        
        # === STOPLOSS LOGIC (Rule 6) - 3 * ATR trailing ===
        stoploss_triggered = False
        
        if in_position and position_side != 0:
            if position_side > 0:
                if close[i] > highest_price:
                    highest_price = close[i]
                stoploss_price = highest_price - 3.0 * atr_14[i]
                if close[i] < stoploss_price:
                    stoploss_triggered = True
            
            if position_side < 0:
                if lowest_price == 0.0 or close[i] < lowest_price:
                    lowest_price = close[i]
                stoploss_price = lowest_price + 3.0 * atr_14[i]
                if close[i] > stoploss_price:
                    stoploss_triggered = True
        
        # === REGIME REVERSAL EXIT ===
        regime_reversal = False
        if in_position and position_side != 0:
            # Long position but regime turns strongly bearish
            if position_side > 0 and regime_bear and price_below_1w_hma:
                regime_reversal = True
            # Short position but regime turns strongly bullish
            if position_side < 0 and regime_bull and price_above_1w_hma:
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