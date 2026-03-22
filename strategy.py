#!/usr/bin/env python3
"""
Experiment #006: 1d Regime-Adaptive Strategy with 1w HMA Trend Bias

Hypothesis: Daily timeframe should excel because:
1. Far fewer trades (10-30/year) = minimal fee drag (0.5-1.5% annually)
2. Less noise than lower TFs = cleaner signals
3. Can survive 2022 crash with proper regime detection and position sizing
4. 1w HMA provides ultra-stable trend bias (changes very rarely)

Strategy combines:
1. 1W HMA trend bias: Only long if price > 1w_HMA, only short if price < 1w_HMA
2. Choppiness Index (CHOP) regime filter: CHOP>61.8 = range (mean revert), 
   CHOP<38.2 = trend (trend follow). This is CRITICAL for bear/range markets.
3. RSI(14) extremes for mean reversion: RSI<30 long, RSI>70 short (in range regime)
4. Donchian(20) breakout for trend following (in trend regime)
5. ATR(14) volatility filter: Skip entries when ATR ratio > 2.5 (panic conditions)
6. Volume confirmation: Volume > 0.8 * 20-day avg
7. Stoploss: 2.5 * ATR trailing stop

Why this should beat Sharpe=0.123:
- Regime-adaptive = works in both bull AND bear/range markets
- 1d has 1/12 the trades of 1h = 12x less fee drag
- 1w HMA is extremely stable (won't flip during 2022 whipsaw)
- CHOP filter prevents trend-following losses in choppy 2022 bottom
- Conservative sizing (0.25-0.30) protects from 77% BTC crash

Timeframe: 1d (REQUIRED for this experiment)
HTF: 1w via mtf_data helper (call ONCE before loop)
Position sizing: 0.25-0.30 discrete, ATR-scaled
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_regime_chop_rsi_1w_hma_donchian_atr_v1"
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

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_choppiness_index(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP > 61.8 = ranging/consolidation (mean reversion regime)
    CHOP < 38.2 = trending (trend following regime)
    Formula: 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    """
    n = period
    atr = calculate_atr(high, low, close, period)
    
    # Sum of ATR over period
    atr_sum = pd.Series(atr).rolling(window=n, min_periods=n).sum().values
    
    # Highest high and lowest low over period
    highest_high = pd.Series(high).rolling(window=n, min_periods=n).max().values
    lowest_low = pd.Series(low).rolling(window=n, min_periods=n).min().values
    
    # Choppiness Index
    price_range = highest_high - lowest_low
    price_range = np.where(price_range == 0, 1e-10, price_range)  # Avoid div by zero
    
    chop = 100 * np.log10(atr_sum / price_range) / np.log10(n)
    chop = np.clip(chop, 0, 100)
    
    return chop

def calculate_rsi(close, period=14):
    """Calculate Relative Strength Index."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.inf)
    rsi = 100 - (100 / (1 + rs))
    
    return rsi.values

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (highest high / lowest low over period)."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    
    upper = high_s.rolling(window=period, min_periods=period).max()
    lower = low_s.rolling(window=period, min_periods=period).min()
    
    return upper.values, lower.values

def calculate_adx(high, low, close, period=14):
    """Calculate Average Directional Index (ADX)."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    # True Range
    tr1 = high_s - low_s
    tr2 = np.abs(high_s - close_s.shift(1))
    tr3 = np.abs(low_s - close_s.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Directional Movement
    plus_dm = high_s.diff()
    minus_dm = -low_s.diff()
    
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
    
    # Smoothed DM and TR (Wilder's smoothing)
    plus_dm_smooth = plus_dm.ewm(span=period, min_periods=period, adjust=False).mean()
    minus_dm_smooth = minus_dm.ewm(span=period, min_periods=period, adjust=False).mean()
    tr_smooth = tr.ewm(span=period, min_periods=period, adjust=False).mean()
    
    # Directional Indicators
    plus_di = 100 * (plus_dm_smooth / tr_smooth.replace(0, np.inf))
    minus_di = 100 * (minus_dm_smooth / tr_smooth.replace(0, np.inf))
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.inf)
    adx = dx.ewm(span=period, min_periods=period, adjust=False).mean()
    
    return adx.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HTF indicators
    hma_1w = calculate_hma(df_1w['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 1d indicators
    atr_14 = calculate_atr(high, low, close, 14)
    chop_14 = calculate_choppiness_index(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, 20)
    adx_14 = calculate_adx(high, low, close, 14)
    
    # Volume SMA for confirmation
    vol_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR ratio for volatility filter (current ATR / 30-day median ATR)
    atr_30_median = pd.Series(atr_14).rolling(window=30, min_periods=30).apply(
        lambda x: np.median(x), raw=True
    ).values
    atr_ratio = atr_14 / np.where(atr_30_median > 0, atr_30_median, 1e-10)
    
    signals = np.zeros(n)
    
    # Base position sizing (Rule 4 - discrete levels, max 0.40)
    BASE_SIZE = 0.28
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_1w_aligned[i]):
            continue
        
        if np.isnan(chop_14[i]):
            continue
        
        if np.isnan(rsi_14[i]):
            continue
        
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            continue
        
        if np.isnan(vol_sma[i]) or vol_sma[i] == 0:
            continue
        
        if np.isnan(atr_ratio[i]):
            continue
        
        # === 1W HMA TREND BIAS (Ultra-stable HTF filter) ===
        bull_bias = close[i] > hma_1w_aligned[i]
        bear_bias = close[i] < hma_1w_aligned[i]
        
        # === CHOPPINESS INDEX REGIME ===
        # CHOP > 61.8 = ranging (mean reversion)
        # CHOP < 38.2 = trending (trend following)
        is_range_regime = chop_14[i] > 55  # Slightly lower threshold for more signals
        is_trend_regime = chop_14[i] < 45  # Slightly higher threshold for more signals
        
        # === VOLATILITY FILTER ===
        # Skip entries during extreme volatility (panic conditions)
        vol_extreme = atr_ratio[i] > 2.5
        
        # === VOLUME CONFIRMATION ===
        volume_confirmed = volume[i] > 0.8 * vol_sma[i]
        
        # === ATR-BASED POSITION SIZING ===
        # Reduce size when volatility is high (protect in crashes)
        size_multiplier = 1.0 / np.clip(atr_ratio[i], 0.5, 2.0)
        current_size = BASE_SIZE * size_multiplier
        current_size = np.clip(current_size, 0.20, 0.32)
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # MODE 1: RANGE REGIME - MEAN REVERSION (RSI extremes)
        # Only enter when price is at extreme AND regime is ranging
        if is_range_regime and not vol_extreme and volume_confirmed:
            # Long: RSI oversold + bullish 1W bias
            if rsi_14[i] < 32 and bull_bias:
                new_signal = current_size
            
            # Short: RSI overbought + bearish 1W bias
            elif rsi_14[i] > 68 and bear_bias:
                new_signal = -current_size
        
        # MODE 2: TREND REGIME - DONCHIAN BREAKOUT
        # Only enter when regime is trending AND breakout occurs
        if is_trend_regime and not vol_extreme and volume_confirmed:
            # Breakout long: price crosses above previous Donchian upper
            if i > 0 and not np.isnan(donchian_upper[i-1]):
                breakout_long = (close[i] > donchian_upper[i-1]) and (close[i-1] <= donchian_upper[i-1])
                if breakout_long and bull_bias:
                    new_signal = current_size
            
            # Breakout short: price crosses below previous Donchian lower
            if i > 0 and not np.isnan(donchian_lower[i-1]):
                breakout_short = (close[i] < donchian_lower[i-1]) and (close[i-1] >= donchian_lower[i-1])
                if breakout_short and bear_bias:
                    new_signal = -current_size
        
        # MODE 3: ADX CONFIRMED TREND (fallback for trend regime)
        if adx_14[i] > 25 and not vol_extreme and volume_confirmed:
            # Long: ADX trending + bullish bias + RSI not overbought
            if bull_bias and rsi_14[i] < 65:
                # Check for Donchian breakout
                if i > 0 and not np.isnan(donchian_upper[i-1]):
                    if close[i] > donchian_upper[i-1]:
                        new_signal = current_size * 0.9
            
            # Short: ADX trending + bearish bias + RSI not oversold
            if bear_bias and rsi_14[i] > 35:
                # Check for Donchian breakout
                if i > 0 and not np.isnan(donchian_lower[i-1]):
                    if close[i] < donchian_lower[i-1]:
                        new_signal = -current_size * 0.9
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        stoploss_triggered = False
        
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest price for long position
                if close[i] > highest_price:
                    highest_price = close[i]
                stoploss_price = highest_price - 2.5 * atr_14[i]
                if close[i] < stoploss_price:
                    stoploss_triggered = True
            
            if position_side < 0:
                # Update lowest price for short position
                if lowest_price == 0.0 or close[i] < lowest_price:
                    lowest_price = close[i]
                stoploss_price = lowest_price + 2.5 * atr_14[i]
                if close[i] > stoploss_price:
                    stoploss_triggered = True
        
        # === REGIME REVERSAL EXIT ===
        # Exit if regime changes against position
        regime_reversal = False
        if in_position and position_side != 0:
            # Exit long if regime becomes strongly trending bearish
            if position_side > 0 and is_trend_regime and bear_bias and adx_14[i] > 25:
                regime_reversal = True
            # Exit short if regime becomes strongly trending bullish
            if position_side < 0 and is_trend_regime and bull_bias and adx_14[i] > 25:
                regime_reversal = True
        
        # Apply stoploss or regime reversal
        if stoploss_triggered or regime_reversal:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                # New entry
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Position flip
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                # Exit position
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_price = 0.0
                lowest_price = 0.0
        
        signals[i] = new_signal
    
    return signals