#!/usr/bin/env python3
"""
Experiment #472: 4h Regime-Adaptive Strategy with Choppiness Index + Multi-TF Trend

Hypothesis: After 471 experiments, the key insight is that regime detection is critical.
Pure trend or pure mean-reversion strategies fail because markets alternate between
trending and ranging. This strategy uses:

1. CHOPPINESS INDEX (CHOP) for regime detection:
   - CHOP > 61.8 = ranging market (use mean reversion)
   - CHOP < 38.2 = trending market (use breakout/trend following)
   - This is the BEST filter for bear/range markets per research

2. WEEKLY HMA(21) for long-term trend bias:
   - Price > 1w HMA = bull bias (prefer longs)
   - Price < 1w HMA = bear bias (prefer shorts)

3. DAILY ADX(14) for trend strength confirmation:
   - ADX > 25 = strong trend (allow breakout entries)
   - ADX < 20 = weak trend (prefer mean reversion)

4. TWO ENTRY MODES based on regime:
   a) RANGE MODE (CHOP > 61.8):
      - Long: Price < BB_lower(20, 2.0) + RSI(14) < 35
      - Short: Price > BB_upper(20, 2.0) + RSI(14) > 65
      - Mean reversion at Bollinger extremes
   
   b) TREND MODE (CHOP < 38.2 + ADX > 25):
      - Long: Price breaks Donchian(10) high + price > 1w HMA
      - Short: Price breaks Donchian(10) low + price < 1w HMA
      - Breakout entries with trend confirmation

5. ATR(14) trailing stop at 2.5x for risk management

6. Position sizing: 0.30 discrete levels (30% capital max)

Why this should work on 4h:
- Choppiness Index is proven to distinguish trend vs range better than ADX alone
- Regime-adaptive logic means we use the right strategy for current market
- Multiple entry modes ensure sufficient trades (>10/year per symbol)
- Weekly HMA + Daily ADX provides robust multi-TF confirmation
- Should work on BTC/ETH/SOL individually (not SOL-biased)

Timeframe: 4h (REQUIRED for experiment #472)
HTF: 1d and 1w via mtf_data helper
Position sizing: 0.30 discrete
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_regime_chop_1d_1w_hma_adaptive_atr_v2"
timeframe = "4h"
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
    """Calculate Hull Moving Average."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_adx(high, low, close, period=14):
    """Calculate Average Directional Index."""
    n = len(close)
    adx = np.full(n, np.nan)
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        plus_move = high[i] - high[i-1]
        minus_move = low[i-1] - low[i]
        
        if plus_move > minus_move and plus_move > 0:
            plus_dm[i] = plus_move
        if minus_move > plus_move and minus_move > 0:
            minus_dm[i] = minus_move
    
    plus_dm_s = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_s = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    tr_s = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    for i in range(period, n):
        if tr_s[i] > 1e-10:
            plus_di = 100 * plus_dm_s[i] / tr_s[i]
            minus_di = 100 * minus_dm_s[i] / tr_s[i]
            di_sum = plus_di + minus_di
            if di_sum > 1e-10:
                dx = 100 * np.abs(plus_di - minus_di) / di_sum
            else:
                dx = 0
        else:
            dx = 0
        
        if i == period:
            adx[i] = dx
        else:
            adx[i] = ((adx[i-1] * (period - 1)) + dx) / period
    
    return adx

def calculate_rsi(close, period=14):
    """Calculate Relative Strength Index."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.inf)
    rsi = 100 - (100 / (1 + rs))
    return rsi.values

def calculate_bollinger(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    middle = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    upper = middle + (std * std_mult)
    lower = middle - (std * std_mult)
    return upper.values, lower.values

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    High CHOP (>61.8) = ranging/choppy market
    Low CHOP (<38.2) = trending market
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    for i in range(period, n):
        sum_atr = np.sum(atr[i-period+1:i+1])
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10 and sum_atr > 0:
            chop[i] = 100 * np.log10(sum_atr / price_range) / np.log10(period)
        else:
            chop[i] = 50
    
    return chop

def calculate_donchian(high, low, period=10):
    """Calculate Donchian Channel."""
    n = len(high)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        upper[i] = high[i-period+1:i+1].max()
        lower[i] = low[i-period+1:i+1].min()
    
    return upper, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HTF indicators
    adx_1d = calculate_adx(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 14)
    hma_1w = calculate_hma(df_1w['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 4h indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    bb_upper, bb_lower = calculate_bollinger(close, 20, 2.0)
    chop = calculate_choppiness(high, low, close, 14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, 10)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE = 0.30
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1w_aligned[i]) or np.isnan(adx_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi[i]) or np.isnan(chop[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            signals[i] = 0.0
            continue
        
        # === REGIME DETECTION ===
        chop_val = chop[i]
        ranging_market = chop_val > 61.8
        trending_market = chop_val < 38.2
        
        # === WEEKLY HMA TREND BIAS ===
        bull_trend_1w = close[i] > hma_1w_aligned[i]
        bear_trend_1w = close[i] < hma_1w_aligned[i]
        
        # === DAILY ADX TREND STRENGTH ===
        adx_1d_val = adx_1d_aligned[i]
        strong_trend = adx_1d_val > 25
        
        # === GENERATE SIGNAL based on regime ===
        new_signal = 0.0
        
        if ranging_market:
            # MEAN REVERSION MODE - looser conditions for more trades
            # Long: Price at BB lower + RSI oversold
            if close[i] <= bb_lower[i] and rsi[i] < 35:
                new_signal = SIZE if bull_trend_1w else SIZE * 0.5
            
            # Short: Price at BB upper + RSI overbought
            elif close[i] >= bb_upper[i] and rsi[i] > 65:
                new_signal = -SIZE if bear_trend_1w else -SIZE * 0.5
        
        elif trending_market and strong_trend:
            # TREND FOLLOWING MODE
            # Long: Donchian breakout + bull weekly bias
            if close[i] > donchian_upper[i-1] and bull_trend_1w:
                new_signal = SIZE
            
            # Short: Donchian breakdown + bear weekly bias
            elif close[i] < donchian_lower[i-1] and bear_trend_1w:
                new_signal = -SIZE
        
        else:
            # MIDDLE REGIME - use RSI extremes for mean reversion
            if rsi[i] < 30 and bull_trend_1w:
                new_signal = SIZE * 0.5
            elif rsi[i] > 70 and bear_trend_1w:
                new_signal = -SIZE * 0.5
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        if in_position and position_side != 0:
            if position_side > 0:
                if close[i] > highest_close:
                    highest_close = close[i]
                stoploss_price = highest_close - 2.5 * atr[i]
                if close[i] < stoploss_price:
                    new_signal = 0.0
            
            if position_side < 0:
                if lowest_close == 0.0 or close[i] < lowest_close:
                    lowest_close = close[i]
                stoploss_price = lowest_close + 2.5 * atr[i]
                if close[i] > stoploss_price:
                    new_signal = 0.0
        
        # === TREND REVERSAL EXIT ===
        if in_position and new_signal != 0.0:
            if position_side > 0 and bear_trend_1w:
                new_signal = 0.0
            if position_side < 0 and bull_trend_1w:
                new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                position_side = np.sign(new_signal)
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                highest_close = 0.0
                lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals