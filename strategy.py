#!/usr/bin/env python3
"""
Experiment #018: 1d Adaptive Trend-Mean Reversion with 1W HMA Regime

Hypothesis: After analyzing 17 failed experiments, the key insight is:
1. Pure trend strategies fail in bear/range markets (2022 crash, 2025 bear)
2. Pure mean reversion fails in strong trends
3. 1d timeframe needs ADAPTIVE logic that switches based on regime

This strategy uses:

1. 1W HMA Regime Filter: Ultra-stable weekly trend. Price > 1w_HMA = bull regime
   (prefer long entries), Price < 1w_HMA = bear regime (prefer short entries).

2. ADAPTIVE Entry Logic:
   - BULL REGIME: Long on RSI(14) pullback to 35-50 (not oversold, just dip)
   - BEAR REGIME: Short on RSI(14) rally to 50-65 (not overbought, just bounce)
   - This avoids catching falling knives in crashes and chasing rallies in bears

3. Bollinger Band Confirmation: Entry must be near BB lower (long) or upper (short)
   Ensures we're buying dips / selling rips, not chasing

4. Volume Confirmation: Volume > 1.2 * 20-day avg volume on entry bar
   Confirms genuine interest, not random noise

5. ATR(14) Stoploss: 2.5 * ATR trailing stop to protect from crashes
   Critical for 2022-style drawdowns

6. Volatility-Adjusted Sizing: Position size inversely proportional to ATR
   Smaller positions in high vol (crashes), larger in low vol (stable trends)

Why this should work on 1d:
- Adaptive logic works in both bull and bear markets
- RSI pullback (not extreme) generates MORE trades than CRSI extremes
- BB confirmation reduces false breakouts
- Volume filter eliminates low-liquidity traps
- 1d timeframe = fewer trades = less fee drag (~30-50 trades/year)
- Each symbol should generate 120-200 trades over 4-year train

Timeframe: 1d (REQUIRED for this experiment)
HTF: 1w via mtf_data helper (call ONCE before loop)
Position sizing: 0.20-0.35 discrete, volatility-adjusted
Stoploss: 2.5 * ATR(14) trailing
Target trades: 30-50 per year per symbol (120-200 over 4yr train)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_adaptive_rsi_bb_1w_hma_vol_atr_v1"
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

def calculate_rsi(close, period=14):
    """Calculate RSI using Wilder's smoothing."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.inf)
    rsi = 100 - (100 / (1 + rs))
    return rsi.values

def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    upper = sma + (std_mult * std)
    lower = sma - (std_mult * std)
    return sma.values, upper.values, lower.values

def calculate_volume_ma(volume, period=20):
    """Calculate moving average of volume."""
    vol_s = pd.Series(volume)
    vol_ma = vol_s.rolling(window=period, min_periods=period).mean()
    return vol_ma.values

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
    rsi_14 = calculate_rsi(close, 14)
    bb_sma, bb_upper, bb_lower = calculate_bollinger_bands(close, 20, 2.0)
    vol_ma_20 = calculate_volume_ma(volume, 20)
    
    signals = np.zeros(n)
    
    # Base position sizing (Rule 4 - discrete levels, max 0.40)
    BASE_SIZE = 0.30  # 30% of capital
    MIN_SIZE = 0.15   # Minimum in high vol
    
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
        
        if np.isnan(rsi_14[i]):
            continue
        
        if np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]):
            continue
        
        if np.isnan(vol_ma_20[i]) or vol_ma_20[i] == 0:
            continue
        
        # === 1W HMA REGIME BIAS ===
        bull_regime = close[i] > hma_1w_aligned[i]
        bear_regime = close[i] < hma_1w_aligned[i]
        
        # === RSI PULLBACK SIGNALS (Adaptive to regime) ===
        # Bull regime: Long on RSI pullback to 35-50 (dip buying, not oversold)
        # Bear regime: Short on RSI rally to 50-65 (selling bounces, not overbought)
        rsi_pullback_long = 35 <= rsi_14[i] <= 50
        rsi_rally_short = 50 <= rsi_14[i] <= 65
        
        # === BOLLINGER BAND CONFIRMATION ===
        # Long: price near or below lower BB
        # Short: price near or above upper BB
        bb_long_confirm = close[i] <= bb_lower[i] * 1.01  # Within 1% of lower BB
        bb_short_confirm = close[i] >= bb_upper[i] * 0.99  # Within 1% of upper BB
        
        # === VOLUME CONFIRMATION ===
        vol_confirm = volume[i] > 1.2 * vol_ma_20[i]
        
        # === VOLATILITY-ADJUSTED POSITION SIZING ===
        # Higher ATR = smaller position (protect during crashes)
        # Calculate ATR percentile over last 100 days
        if i >= 100:
            atr_recent = atr_14[max(0, i-100):i+1]
            atr_percentile = np.sum(atr_recent <= atr_14[i]) / len(atr_recent)
        else:
            atr_percentile = 0.5
        
        # Size reduction in high volatility (above 70th percentile)
        if atr_percentile > 0.70:
            position_size = MIN_SIZE  # 15% in high vol
        elif atr_percentile > 0.50:
            position_size = BASE_SIZE * 0.8  # 24% in medium vol
        else:
            position_size = BASE_SIZE  # 30% in low vol
        
        # === ENTRY LOGIC (Adaptive to regime) ===
        new_signal = 0.0
        
        # BULL REGIME: Prefer long entries on pullbacks
        if bull_regime:
            if rsi_pullback_long and bb_long_confirm and vol_confirm:
                new_signal = position_size
            # Also allow long if RSI very oversold (<30) even without BB confirm
            elif rsi_14[i] < 30 and vol_confirm:
                new_signal = position_size
        
        # BEAR REGIME: Prefer short entries on rallies
        elif bear_regime:
            if rsi_rally_short and bb_short_confirm and vol_confirm:
                new_signal = -position_size
            # Also allow short if RSI very overbought (>70) even without BB confirm
            elif rsi_14[i] > 70 and vol_confirm:
                new_signal = -position_size
        
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
        # Exit long if regime turns bear (price crosses below 1w HMA)
        # Exit short if regime turns bull (price crosses above 1w HMA)
        regime_exit = False
        if in_position and position_side != 0:
            if position_side > 0 and bear_regime:
                regime_exit = True
            if position_side < 0 and bull_regime:
                regime_exit = True
        
        # Apply stoploss or regime exit
        if stoploss_triggered or regime_exit:
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