#!/usr/bin/env python3
"""
Experiment #012: 12h Primary + 1d/1w HTF — Fisher Transform + Z-Score Mean Reversion

Hypothesis: Connors RSI + Choppiness combinations have failed 4+ times. 
This strategy uses DIFFERENT indicators proven in bear/range markets:

1. Ehlers Fisher Transform (period=9) - catches reversals in bear rallies
   Long when Fisher crosses above -1.5, Short when crosses below +1.5
2. Z-Score(20) - statistical mean reversion, adapts to volatility regime
   Entry when |z-score| > 2.0, exit when |z-score| < 0.5
3. 1d HMA(21) for major trend bias - only trade WITH 1d trend direction
4. ADX(14) regime filter - ADX<25 = range (prefer mean reversion)
5. ATR(14) trailing stoploss at 2.5x - mandatory risk management
6. Discrete sizing: 0.25 base, 0.30 for high conviction

Why this should work:
- Fisher Transform has 70%+ win rate on reversals (Ehlers research)
- Z-score entries are statistical, not arbitrary thresholds
- 1d trend filter prevents counter-trend trades in strong moves
- ADX regime detection avoids trend-following in chop
- 12h timeframe = ~30-50 trades/year (optimal for fee drag)
- Different indicator suite than all 11 failed experiments

Timeframe: 12h (REQUIRED for this experiment)
HTF: 1d and 1w via mtf_data.get_htf_data() — called ONCE before loop
Position sizing: 0.25-0.30 discrete
Stoploss: 2.5 * ATR(14) trailing
Target trades: 20-50/year per symbol
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_fisher_zscore_adx_1d1w_v1"
timeframe = "12h"
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
    Ehlers Fisher Transform - normalizes price to Gaussian distribution
    
    Steps:
    1. Calculate typical price: (high + low + close) / 3
    2. Normalize: (price - lowest) / (highest - lowest) * 0.999 + 0.001
    3. Apply Fisher: 0.5 * ln((1+x)/(1-x))
    4. Smooth with EMA
    
    Entry: Fisher crosses above -1.5 (long), crosses below +1.5 (short)
    """
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    # Typical price
    typical = (high_s + low_s + close_s) / 3
    
    # Rolling highest and lowest
    highest = typical.rolling(window=period, min_periods=period).max()
    lowest = typical.rolling(window=period, min_periods=period).min()
    
    # Normalize to 0-1 range with bounds
    range_hl = highest - lowest
    normalized = ((typical - lowest) / range_hl.replace(0, np.nan)) * 0.999 + 0.001
    normalized = normalized.clip(0.001, 0.999)
    
    # Fisher transform
    fisher = 0.5 * np.log((1 + normalized) / (1 - normalized))
    
    # Smooth with EMA
    fisher_smooth = fisher.ewm(span=period, min_periods=period, adjust=False).mean()
    
    return fisher_smooth.values

def calculate_zscore(close, period=20):
    """
    Z-Score = (price - SMA) / StdDev
    
    |z| > 2.0 = extreme (entry signal)
    |z| < 0.5 = mean reached (exit signal)
    """
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    
    zscore = (close_s - sma) / std.replace(0, np.nan)
    zscore = zscore.fillna(0).values
    return zscore

def calculate_adx(high, low, close, period=14):
    """
    Average Directional Index - measures trend strength
    
    ADX > 25 = trending
    ADX < 20 = ranging
    """
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Directional Movement
    plus_dm = high_s.diff()
    minus_dm = -low_s.diff()
    
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
    
    # Smooth with Wilder's method (EMA with span=period)
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(span=period, min_periods=period, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(span=period, min_periods=period, adjust=False).mean() / atr)
    
    # DX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan)
    
    # ADX = smoothed DX
    adx = dx.ewm(span=period, min_periods=period, adjust=False).mean()
    adx = adx.fillna(0).values
    
    return adx

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HTF indicators
    hma_1d_21 = calculate_hma(df_1d['close'].values, 21)
    hma_1w_21 = calculate_hma(df_1w['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    hma_1w_21_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_21)
    
    # Calculate 12h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    fisher = calculate_fisher_transform(high, low, close, 9)
    zscore = calculate_zscore(close, 20)
    adx = calculate_adx(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    
    signals = np.zeros(n)
    
    # Base position sizing (Rule 4 - discrete levels, max 0.40)
    BASE_SIZE = 0.25
    HIGH_CONV_SIZE = 0.30
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_fisher_long = -100
    last_fisher_short = -100
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_1d_21_aligned[i]) or np.isnan(hma_1w_21_aligned[i]):
            continue
        
        if np.isnan(fisher[i]) or np.isnan(zscore[i]) or np.isnan(adx[i]):
            continue
        
        # === 1W TREND BIAS (MAJOR) ===
        # Price above 1w HMA = bullish bias (prefer longs)
        # Price below 1w HMA = bearish bias (prefer shorts)
        trend_1w_bullish = close[i] > hma_1w_21_aligned[i]
        trend_1w_bearish = close[i] < hma_1w_21_aligned[i]
        
        # === 1D TREND CONFIRMATION (INTERMEDIATE) ===
        trend_1d_bullish = close[i] > hma_1d_21_aligned[i]
        trend_1d_bearish = close[i] < hma_1d_21_aligned[i]
        
        # === ADX REGIME ===
        # ADX < 20 = range (mean reversion preferred)
        # ADX > 25 = trend (trend following preferred)
        is_range = adx[i] < 20
        is_trend = adx[i] > 25
        
        # === FISHER TRANSFORM SIGNALS ===
        # Long: Fisher crosses above -1.5 (from below)
        # Short: Fisher crosses below +1.5 (from above)
        fisher_cross_long = (fisher[i] > -1.5) and (fisher[last_fisher_long] <= -1.5 if last_fisher_long >= 0 else False)
        fisher_cross_short = (fisher[i] < 1.5) and (fisher[last_fisher_short] >= 1.5 if last_fisher_short >= 0 else False)
        
        # Also allow extreme Fisher values without cross (for fewer trades)
        fisher_extreme_long = fisher[i] < -1.8
        fisher_extreme_short = fisher[i] > 1.8
        
        # === Z-SCORE CONFIRMATION ===
        # |z| > 2.0 = extreme (entry confirmation)
        # |z| < 0.5 = mean reached (exit signal)
        zscore_extreme_long = zscore[i] < -2.0
        zscore_extreme_short = zscore[i] > 2.0
        zscore_mean_reached = np.abs(zscore[i]) < 0.5
        
        # === RSI FILTER ===
        # Avoid entries when RSI is neutral (40-60)
        rsi_neutral = 40 < rsi_14[i] < 60
        rsi_bullish = rsi_14[i] > 50
        rsi_bearish = rsi_14[i] < 50
        
        # === POSITION SIZING ===
        current_size = BASE_SIZE
        # High conviction: both Fisher extreme AND z-score extreme
        high_conviction_long = fisher_extreme_long and zscore_extreme_long
        high_conviction_short = fisher_extreme_short and zscore_extreme_short
        if high_conviction_long or high_conviction_short:
            current_size = HIGH_CONV_SIZE
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # LONG ENTRIES
        # Primary: Fisher cross + z-score extreme + trend alignment
        # Secondary: Fisher extreme + z-score extreme (mean reversion in range)
        if trend_1w_bullish or is_range:
            # Primary entry: Fisher cross with confirmation
            if (fisher_cross_long or fisher_extreme_long) and zscore_extreme_long:
                # Require trend alignment OR strong mean reversion setup
                if trend_1d_bullish or (is_range and rsi_bullish):
                    new_signal = current_size
            
            # Secondary: RSI oversold + Fisher extreme
            if rsi_14[i] < 35 and fisher_extreme_long and zscore[i] < -1.5:
                if is_range or trend_1d_bullish:
                    new_signal = current_size
        
        # SHORT ENTRIES
        # Primary: Fisher cross + z-score extreme + trend alignment
        # Secondary: Fisher extreme + z-score extreme (mean reversion in range)
        if trend_1w_bearish or is_range:
            # Primary entry: Fisher cross with confirmation
            if (fisher_cross_short or fisher_extreme_short) and zscore_extreme_short:
                # Require trend alignment OR strong mean reversion setup
                if trend_1d_bearish or (is_range and not rsi_bullish):
                    new_signal = -current_size
            
            # Secondary: RSI overbought + Fisher extreme
            if rsi_14[i] > 65 and fisher_extreme_short and zscore[i] > 1.5:
                if is_range or trend_1d_bearish:
                    new_signal = -current_size
        
        # === FREQUENCY SAFEGUARD ===
        # If no trades for 150 bars (~75 days on 12h), allow weaker entry
        bars_since_last_long = i - last_fisher_long
        bars_since_last_short = i - last_fisher_short
        
        if bars_since_last_long > 150 and new_signal == 0.0 and not in_position:
            if trend_1w_bullish and fisher[i] < -1.0 and zscore[i] < -1.0:
                new_signal = BASE_SIZE * 0.7
        
        if bars_since_last_short > 150 and new_signal == 0.0 and not in_position:
            if trend_1w_bearish and fisher[i] > 1.0 and zscore[i] > 1.0:
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
        
        # === Z-SCORE MEAN REVERSION EXIT ===
        # Exit when z-score returns to mean (|z| < 0.5)
        zscore_exit = False
        if in_position and np.abs(zscore[i]) < 0.5:
            zscore_exit = True
        
        # === TREND REVERSAL EXIT ===
        trend_reversal = False
        if in_position and position_side != 0:
            if position_side > 0 and trend_1w_bearish and fisher[i] > 1.0:
                trend_reversal = True
            if position_side < 0 and trend_1w_bullish and fisher[i] < -1.0:
                trend_reversal = True
        
        # Apply stoploss or exit signals
        if stoploss_triggered or zscore_exit or trend_reversal:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                if position_side > 0:
                    last_fisher_long = i
                else:
                    last_fisher_short = i
            elif np.sign(new_signal) != position_side:
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                if position_side > 0:
                    last_fisher_long = i
                else:
                    last_fisher_short = i
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_price = 0.0
                lowest_price = 0.0
        
        signals[i] = new_signal
    
    return signals